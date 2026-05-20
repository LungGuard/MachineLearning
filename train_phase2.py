"""
Phase 2 regression training script — ResNetNoduleModel (pretrained backbone).

Usage:
    python train_phase2.py [options]

Uses ResNet18 with ImageNet pretrained weights, partial backbone freezing,
and a deep regression head. Targets: MAE 0.50-0.75, RMSE 0.70-1.00, R2 0.40-0.65.
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

from DetectionModel.src.models.resnet_nodule_model import ResNetNoduleModel
from DetectionModel.src.data_modules.regression_dataset_module import RegressionDataModule
from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import ModelStage, Loss


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Train ResNetNoduleModel (pretrained backbone)")
    p.add_argument("--iteration",         type=int,   default=1,     help="Iteration label")
    p.add_argument("--lr",                type=float, default=1e-4,  help="Head learning rate")
    p.add_argument("--backbone_lr_factor", type=float, default=0.1,  help="Backbone LR = lr * factor")
    p.add_argument("--wd",                type=float, default=1e-4,  help="Weight decay (AdamW)")
    p.add_argument("--dropout",           type=float, default=0.3,   help="Dropout in head")
    p.add_argument("--head_hidden",       type=int,   default=256,   help="Head hidden size (0=linear only)")
    p.add_argument("--crop_size",         type=int,   default=224,   help="Nodule crop size (px)")
    p.add_argument("--batch_size",        type=int,   default=32,    help="Batch size")
    p.add_argument("--max_epochs",        type=int,   default=150,   help="Max training epochs")
    p.add_argument("--patience",          type=int,   default=30,    help="EarlyStopping patience")
    p.add_argument("--plat_patience",     type=int,   default=10,    help="ReduceLROnPlateau patience")
    p.add_argument("--plat_factor",       type=float, default=0.5,   help="ReduceLROnPlateau factor")
    p.add_argument("--freeze_backbone",   action="store_true", default=True,
                   help="Freeze backbone, unfreeze layer3+layer4 (default)")
    p.add_argument("--head_only",         action="store_true", default=False,
                   help="Freeze ALL backbone layers (head only training)")
    p.add_argument("--augment",           action="store_true", default=True,
                   help="Enable training augmentation")
    p.add_argument("--no_augment",        action="store_true", default=False,
                   help="Disable training augmentation")
    p.add_argument("--no_scale",          action="store_true", default=False,
                   help="Disable StandardScaler on targets")
    p.add_argument("--grad_clip",         type=float, default=1.0,   help="Gradient clip norm (0=off)")
    p.add_argument("--resume",            action="store_true", default=False,
                   help="Resume from best checkpoint")
    return p.parse_args()


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


def build_model(args) -> ResNetNoduleModel:
    return ResNetNoduleModel(
        learning_rate=args.lr,
        backbone_lr_factor=args.backbone_lr_factor,
        weight_decay=args.wd,
        dropout_p=args.dropout,
        head_hidden=args.head_hidden,
        freeze_backbone=args.freeze_backbone,
        head_only=args.head_only,
        plat_patience=args.plat_patience,
        plat_factor=args.plat_factor,
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
    if not hasattr(pathlib, 'WindowsPath'):
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, nn.MSELoss])
    except Exception:
        pass

    args = parse_args()
    augment = args.augment and not args.no_augment
    scale_targets = not args.no_scale

    dm    = build_datamodule(args.crop_size, args.batch_size, augment, scale_targets)
    model = build_model(args)

    callbacks = build_callbacks(args.patience)

    run_name = (
        f"p2_iter{args.iteration}"
        f"_lr{args.lr}_blrf{args.backbone_lr_factor}"
        f"_wd{args.wd}_drop{args.dropout}"
        f"_head{args.head_hidden}"
        f"{'_headonly' if args.head_only else '_partial'}"
    )

    trainer_kwargs = dict(
        accelerator='auto',
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=str(RegressionModelConstants.LOG_DIR),
            name="regression_phase2",
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

    print(f"\n[Phase2] Iteration {args.iteration}")
    print(f"  freeze_backbone={args.freeze_backbone}, head_only={args.head_only}")
    print(f"  lr={args.lr} (backbone×{args.backbone_lr_factor}), wd={args.wd}, dropout={args.dropout}")
    print(f"  head_hidden={args.head_hidden}, crop={args.crop_size}px, batch={args.batch_size}")
    print(f"  max_epochs={args.max_epochs}, patience={args.patience}")
    print(f"  augment={augment}, scale_targets={scale_targets}")
    print(f"  grad_clip={args.grad_clip}, resume={ckpt_path or 'no'}\n")

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

    print("\n=== Final validation metrics ===")
    val_result = trainer.validate(model=model, datamodule=dm, verbose=True)
    print(val_result)


if __name__ == "__main__":
    main()
