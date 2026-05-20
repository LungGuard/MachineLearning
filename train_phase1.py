"""
Phase 1 regression training script — NoduleFeaturesModel (custom CNN).

Usage:
    python train_phase1.py [options]

Tunes NoduleFeaturesModel depth, filter sizes, dropout, lr, wd on the
LIDC-IDRI nodule crop dataset (64x64 crops). Logs to TensorBoard.
Checkpoints land in DetectionModel/src/model_checkpoints/.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import lightning.pytorch.callbacks as LightningCallbacks
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import pathlib

from DetectionModel.src.models.regression_model import NoduleFeaturesModel
from DetectionModel.src.data_modules.regression_dataset_module import RegressionDataModule
from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import ModelStage, Loss

try:
    Accelerator = __import__('common.constants', fromlist=['Accelerator']).Accelerator
except Exception:
    class Accelerator:
        AUTO = 'auto'


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Train NoduleFeaturesModel (custom CNN)")
    p.add_argument("--iteration",    type=int,   default=1,     help="Iteration label for logging")
    p.add_argument("--lr",           type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--wd",           type=float, default=1e-4,  help="Weight decay")
    p.add_argument("--dropout",      type=float, default=0.3,   help="Dropout before output layer")
    p.add_argument("--crop_size",    type=int,   default=64,    help="Nodule crop size (px)")
    p.add_argument("--batch_size",   type=int,   default=64,    help="Batch size")
    p.add_argument("--max_epochs",   type=int,   default=120,   help="Max training epochs")
    p.add_argument("--patience",     type=int,   default=20,    help="EarlyStopping patience")
    p.add_argument("--augment",      action="store_true", default=False,
                   help="Enable training augmentation")
    p.add_argument("--no_scale",     action="store_true", default=False,
                   help="Disable StandardScaler on targets")
    p.add_argument("--conv_layers",  type=str,   default="32,64,128",
                   help="Comma-separated conv channel sizes (e.g. 32,64,128,256)")
    p.add_argument("--dense_layers", type=str,   default="128,64",
                   help="Comma-separated dense hidden sizes (e.g. 256,128,64)")
    p.add_argument("--resume",       action="store_true", default=False,
                   help="Resume from best checkpoint if it exists")
    p.add_argument("--grad_clip",    type=float, default=0.0,
                   help="Gradient clipping value (0=disabled)")
    p.add_argument("--plat_patience", type=int, default=8,
                   help="ReduceLROnPlateau patience before LR halving")
    return p.parse_args()


def parse_layers(s: str) -> tuple:
    return tuple(int(x.strip()) for x in s.split(","))


def build_datamodule(crop_size: int, batch_size: int, augment: bool, scale_targets: bool) -> RegressionDataModule:
    return RegressionDataModule(
        metadata_csv=DatasetConstants.DATASET_DIR,
        dataset_root=DatasetConstants.PROJECT_ROOT,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        augment_train=augment,
        scale_targets=scale_targets,
    )


def build_model(lr: float, wd: float, dropout: float,
                conv_layers: tuple, dense_layers: tuple,
                crop_size: int, plat_patience: int = 8) -> NoduleFeaturesModel:
    model = NoduleFeaturesModel(
        input_shape=(3, crop_size, crop_size),
        learning_rate=lr,
        weight_decay=wd,
        dropout_p=dropout,
        conv_layers_channels=conv_layers,
        dense_layers_channels=dense_layers,
        loss_fn=nn.MSELoss(),
    )
    model._plat_patience = plat_patience
    return model


def build_callbacks(patience: int) -> list:
    val_loss_key = Loss.DEFAULT.get_variant(ModelStage.VAL)
    return [
        LightningCallbacks.EarlyStopping(
            monitor=val_loss_key,
            patience=patience,
            mode="min",
            verbose=True,
        ),
        LightningCallbacks.ModelCheckpoint(
            dirpath=RegressionModelConstants.CHECKPOINT_DIR,
            filename=RegressionModelConstants.BEST_MODEL_CHECKPOINT_NAME,
            monitor=val_loss_key,
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
        LightningCallbacks.OnExceptionCheckpoint(
            dirpath=RegressionModelConstants.CHECKPOINT_DIR,
            filename=RegressionModelConstants.EXCEPTION_CHECKPOINT_FILE_NAME,
        ),
        LightningCallbacks.LearningRateMonitor(logging_interval="epoch"),
    ]


def main():
    if not hasattr(pathlib, 'WindowsPath'):
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, nn.MSELoss])
    except Exception:
        pass

    args = parse_args()

    conv_layers  = parse_layers(args.conv_layers)
    dense_layers = parse_layers(args.dense_layers)

    dm    = build_datamodule(args.crop_size, args.batch_size, args.augment, not args.no_scale)
    model = build_model(args.lr, args.wd, args.dropout, conv_layers, dense_layers, args.crop_size, args.plat_patience)

    callbacks = build_callbacks(args.patience)

    run_name = (
        f"p1_iter{args.iteration}"
        f"_conv[{args.conv_layers.replace(',','-')}]"
        f"_dense[{args.dense_layers.replace(',','-')}]"
        f"_lr{args.lr}_wd{args.wd}_drop{args.dropout}"
    )

    trainer_kwargs = dict(
        accelerator='auto',
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=str(RegressionModelConstants.LOG_DIR),
            name="regression_phase1",
            version=run_name,
        ),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    if args.grad_clip > 0:
        trainer_kwargs['gradient_clip_val'] = args.grad_clip
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    trainer = L.Trainer(**trainer_kwargs)

    best_ckpt = (
        RegressionModelConstants.CHECKPOINT_DIR
        / f"{RegressionModelConstants.BEST_MODEL_CHECKPOINT_NAME}.ckpt"
    )
    ckpt_path = str(best_ckpt) if (args.resume and best_ckpt.exists()) else None

    print(f"\n[Phase1] Iteration {args.iteration}")
    print(f"  conv_layers  = {conv_layers}")
    print(f"  dense_layers = {dense_layers}")
    print(f"  lr={args.lr}, wd={args.wd}, dropout={args.dropout}")
    print(f"  crop={args.crop_size}px, batch={args.batch_size}, max_epochs={args.max_epochs}")
    print(f"  augment={args.augment}, scale_targets={not args.no_scale}")
    print(f"  resume={ckpt_path or 'no'}\n")

    if ckpt_path:
        print(f"[Phase1] Resuming from: {ckpt_path}")
    else:
        print("[Phase1] Starting fresh")

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

    print("\n=== Final validation metrics ===")
    val_result = trainer.validate(model=model, datamodule=dm, verbose=True)
    print(val_result)


if __name__ == "__main__":
    main()
