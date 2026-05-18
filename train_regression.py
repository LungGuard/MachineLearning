"""
Standalone regression model training script.

Usage:
    python train_regression.py [--iteration N] [--lr LR] [--wd WD]

Trains ResNetNoduleModel on the LIDC-IDRI nodule crop dataset and logs
metrics to TensorBoard.  Checkpoints land in
DetectionModel/src/model_checkpoints/.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import lightning.pytorch.callbacks as LightningCallbacks
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
import pathlib
import torch

from DetectionModel.src.models.resnet_nodule_model import ResNetNoduleModel
from DetectionModel.src.data_modules.regression_dataset_module import RegressionDataModule
from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import ModelStage, Accelerator, Loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration",      type=int,   default=1,    help="Iteration label for logging")
    p.add_argument("--lr",             type=float, default=2e-4, help="Head learning rate")
    p.add_argument("--wd",             type=float, default=1e-3, help="Weight decay")
    p.add_argument("--dropout",        type=float, default=0.3,  help="Dropout probability")
    p.add_argument("--crop_size",      type=int,   default=224,  help="Nodule crop size (px)")
    p.add_argument("--batch_size",     type=int,   default=32,   help="Batch size")
    p.add_argument("--max_epochs",     type=int,   default=150,  help="Max training epochs")
    p.add_argument("--patience",       type=int,   default=30,   help="EarlyStopping patience")
    p.add_argument("--freeze_backbone",action="store_true", default=True,
                   help="Freeze ResNet layers 1-3 (layer4 is trainable)")
    p.add_argument("--head_only",      action="store_true", default=False,
                   help="Freeze ALL backbone layers, only train head (~400K params)")
    p.add_argument("--mixup_alpha",    type=float, default=0.0,  help="Mixup alpha (0=disabled)")
    p.add_argument("--resume",         action="store_true", default=False,
                   help="Resume from best checkpoint if it exists")
    p.add_argument("--warm_start",     type=str, default=None,
                   help="Load weights from this checkpoint file (architecture may differ)")
    p.add_argument("--head_hidden",    type=int, default=256,
                   help="Hidden size for regression head (0=linear only, default=256)")
    p.add_argument("--cosine_T0",      type=int, default=0,
                   help="CosineAnnealingWarmRestarts T_0 (0=use ReduceLROnPlateau)")
    p.add_argument("--full_finetune",  action="store_true", default=False,
                   help="Unfreeze ALL backbone layers with layerwise LR decay")
    p.add_argument("--weighted_loss",  action="store_true", default=False,
                   help="Use per-feature weighted MSE (down-weights near-constant features)")
    return p.parse_args()


def build_datamodule(crop_size: int, batch_size: int) -> RegressionDataModule:
    return RegressionDataModule(
        metadata_csv=DatasetConstants.DATASET_DIR,
        dataset_root=DatasetConstants.PROJECT_ROOT,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )


def build_model(lr: float, wd: float, dropout: float,
                freeze_backbone: bool, head_only: bool,
                mixup_alpha: float = 0.0,
                head_hidden: int = 256,
                cosine_T0: int = 0,
                full_finetune: bool = False,
                weighted_loss: bool = False) -> ResNetNoduleModel:
    return ResNetNoduleModel(
        learning_rate=lr,
        weight_decay=wd,
        dropout_p=dropout,
        freeze_backbone=freeze_backbone,
        head_only=head_only,
        full_finetune=full_finetune,
        mixup_alpha=mixup_alpha,
        head_hidden=head_hidden,
        cosine_T0=cosine_T0,
        use_weighted_loss=weighted_loss,
    )


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
    import pathlib as _pathlib
    if not hasattr(_pathlib, 'WindowsPath'):
        _pathlib.WindowsPath = _pathlib.PosixPath
    try:
        torch.serialization.add_safe_globals([_pathlib.WindowsPath, _pathlib.PosixPath, nn.MSELoss])
    except Exception:
        pass

    args = parse_args()

    dm     = build_datamodule(args.crop_size, args.batch_size)
    model  = build_model(args.lr, args.wd, args.dropout,
                         args.freeze_backbone, args.head_only, args.mixup_alpha,
                         args.head_hidden, args.cosine_T0, args.full_finetune,
                         args.weighted_loss)
    callbacks = build_callbacks(args.patience)

    run_name = f"iter{args.iteration}_lr{args.lr}_wd{args.wd}_drop{args.dropout}"

    trainer = L.Trainer(
        accelerator=Accelerator.AUTO,
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=str(RegressionModelConstants.LOG_DIR),
            name="regression_resnet",
            version=run_name,
        ),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    best_ckpt = (
        RegressionModelConstants.CHECKPOINT_DIR
        / f"{RegressionModelConstants.BEST_MODEL_CHECKPOINT_NAME}.ckpt"
    )
    ckpt_path = str(best_ckpt) if (args.resume and best_ckpt.exists()) else None

    if args.warm_start and Path(args.warm_start).exists():
        prior_ckpt = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        model.load_state_dict(prior_ckpt['state_dict'], strict=False)
        print(f"[train_regression] Warm-started from: {args.warm_start}")

    if ckpt_path:
        print(f"[train_regression] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[train_regression] Starting fresh (no checkpoint)")

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

    print("\n=== Final validation metrics ===")
    val_result = trainer.validate(model=model, datamodule=dm, verbose=True)
    print(val_result)


if __name__ == "__main__":
    main()
