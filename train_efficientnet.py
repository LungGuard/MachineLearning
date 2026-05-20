"""
Phase 2 EfficientNet-B0 regression training script.

Usage:
    python train_efficientnet.py [options]

EfficientNet-B0 with partial unfreezing (features[unfreeze_from:]).
Feature vector: 1280-dim. Targets: MAE 0.50-0.75, RMSE 0.70-1.00, R2 0.40-0.65.
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

from DetectionModel.src.models.efficientnet_nodule_model import EfficientNetNoduleModel
from DetectionModel.src.models.weighted_regression_loss import WeightedMSELoss, WeightedHuberLoss
from DetectionModel.src.data_modules.regression_dataset_module import RegressionDataModule
from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import ModelStage, Loss


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Train EfficientNet-B0 backbone")
    p.add_argument("--iteration",         type=int,   default=1,     help="Iteration label")
    p.add_argument("--lr",                type=float, default=5e-4,  help="Head learning rate")
    p.add_argument("--backbone_lr_factor", type=float, default=0.1,  help="Backbone LR = lr * factor")
    p.add_argument("--wd",                type=float, default=5e-3,  help="Weight decay (AdamW)")
    p.add_argument("--dropout",           type=float, default=0.3,   help="Dropout in head")
    p.add_argument("--head_hidden",       type=int,   default=256,   help="Head hidden size (0=linear only)")
    p.add_argument("--unfreeze_from",     type=int,   default=7,     help="Unfreeze EfficientNet blocks from this index (0-8)")
    p.add_argument("--crop_size",         type=int,   default=224,   help="Crop size (px)")
    p.add_argument("--batch_size",        type=int,   default=32,    help="Batch size")
    p.add_argument("--max_epochs",        type=int,   default=150,   help="Max training epochs")
    p.add_argument("--patience",          type=int,   default=30,    help="EarlyStopping patience")
    p.add_argument("--plat_patience",     type=int,   default=10,    help="ReduceLROnPlateau patience")
    p.add_argument("--plat_factor",       type=float, default=0.5,   help="ReduceLROnPlateau factor")
    p.add_argument("--augment",           action="store_true", default=True)
    p.add_argument("--no_augment",        action="store_true", default=False)
    p.add_argument("--no_scale",          action="store_true", default=False)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--resume",            action="store_true", default=False)
    p.add_argument("--warm_start",        type=str,   default=None,
                   help="Load weights from this checkpoint (warm start, may differ in frozen/unfrozen state)")
    p.add_argument("--loss",              type=str,   default="mse",
                   choices=["mse", "huber", "wmse", "whuber"],
                   help="Loss function: mse | huber | wmse (weighted) | whuber (weighted huber)")
    p.add_argument("--huber_delta",       type=float, default=1.0,
                   help="Delta for huber/whuber loss (default 1.0)")
    p.add_argument("--cosine_T0",         type=int,   default=0,
                   help="CosineAnnealingWarmRestarts T_0 (0=use ReduceLROnPlateau)")
    p.add_argument("--cosine_T_mult",     type=int,   default=2,
                   help="CosineAnnealingWarmRestarts T_mult (default 2)")
    return p.parse_args()


def build_datamodule(crop_size, batch_size, augment, scale_targets):
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


def _make_loss_fn(args) -> nn.Module:
    if args.loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta)
    if args.loss == "wmse":
        return WeightedMSELoss()
    if args.loss == "whuber":
        return WeightedHuberLoss(delta=args.huber_delta)
    return nn.MSELoss()


def build_model(args) -> EfficientNetNoduleModel:
    loss_fn = _make_loss_fn(args)
    return EfficientNetNoduleModel(
        learning_rate=args.lr,
        backbone_lr_factor=args.backbone_lr_factor,
        weight_decay=args.wd,
        dropout_p=args.dropout,
        head_hidden=args.head_hidden,
        unfreeze_from=args.unfreeze_from,
        plat_patience=args.plat_patience,
        plat_factor=args.plat_factor,
        cosine_T0=args.cosine_T0,
        cosine_T_mult=args.cosine_T_mult,
        loss_fn=loss_fn,
    )


def build_callbacks(patience: int, run_tag: str) -> list:
    val_loss_key = Loss.DEFAULT.get_variant(ModelStage.VAL)
    ckpt_name = f"best_reg_model_effnet_{run_tag}"
    return [
        LightningCallbacks.EarlyStopping(
            monitor=val_loss_key,
            patience=patience,
            mode="min",
            verbose=True,
        ),
        LightningCallbacks.ModelCheckpoint(
            dirpath=RegressionModelConstants.CHECKPOINT_DIR,
            filename=ckpt_name,
            monitor=val_loss_key,
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
        LightningCallbacks.OnExceptionCheckpoint(
            dirpath=RegressionModelConstants.CHECKPOINT_DIR,
            filename="exception_effnet_checkpoint",
        ),
        LightningCallbacks.LearningRateMonitor(logging_interval="epoch"),
    ]


def main():
    if not hasattr(pathlib, 'WindowsPath'):
        pathlib.WindowsPath = pathlib.PosixPath
    from DetectionModel.src.models.weighted_regression_loss import WeightedMSELoss, WeightedHuberLoss
    try:
        torch.serialization.add_safe_globals([
            pathlib.WindowsPath, pathlib.PosixPath,
            nn.MSELoss, nn.HuberLoss, WeightedMSELoss, WeightedHuberLoss,
        ])
    except Exception:
        pass

    args = parse_args()
    augment = args.augment and not args.no_augment
    scale_targets = not args.no_scale

    loss_tag = args.loss if args.loss == "mse" else f"{args.loss}_d{args.huber_delta}" if "huber" in args.loss else args.loss
    run_tag = f"iter{args.iteration}_uf{args.unfreeze_from}_lr{args.lr}_wd{args.wd}"
    run_name = f"effnet_{run_tag}_drop{args.dropout}_head{args.head_hidden}_{loss_tag}"

    dm    = build_datamodule(args.crop_size, args.batch_size, augment, scale_targets)
    model = build_model(args)
    callbacks = build_callbacks(args.patience, run_tag)

    trainer_kwargs = dict(
        accelerator='auto',
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=str(RegressionModelConstants.LOG_DIR),
            name="regression_efficientnet",
            version=run_name,
        ),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    if args.grad_clip > 0:
        trainer_kwargs['gradient_clip_val'] = args.grad_clip
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    trainer = L.Trainer(**trainer_kwargs)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[EfficientNet Iter {args.iteration}]")
    print(f"  unfreeze_from={args.unfreeze_from} (blocks {args.unfreeze_from}-8 + head)")
    print(f"  lr={args.lr} (backbone×{args.backbone_lr_factor}), wd={args.wd}, dropout={args.dropout}")
    print(f"  head_hidden={args.head_hidden}, crop={args.crop_size}px, batch={args.batch_size}")
    print(f"  loss={loss_tag}, total={total:,}, trainable={trainable:,}")
    print(f"  augment={augment}, scale_targets={scale_targets}, grad_clip={args.grad_clip}\n")

    best_ckpt = RegressionModelConstants.CHECKPOINT_DIR / f"best_reg_model_effnet_{run_tag}.ckpt"
    ckpt_path = str(best_ckpt) if (args.resume and best_ckpt.exists()) else None

    if args.warm_start and Path(args.warm_start).exists():
        prior_ckpt = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        current_sd = model.state_dict()
        compatible = {
            k: v for k, v in prior_ckpt['state_dict'].items()
            if k in current_sd and current_sd[k].shape == v.shape
        }
        skipped = [k for k in prior_ckpt['state_dict'] if k not in compatible]
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if skipped:   print(f"[warm_start] Skipped (shape mismatch): {skipped[:5]}")
        if missing:   print(f"[warm_start] Missing keys: {missing[:5]}")
        if unexpected: print(f"[warm_start] Unexpected keys: {unexpected[:5]}")
        print(f"[warm_start] Loaded {len(compatible)}/{len(prior_ckpt['state_dict'])} tensors from: {args.warm_start}")

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

    print("\n=== Final validation metrics ===")
    val_result = trainer.validate(model=model, datamodule=dm, verbose=True)
    print(val_result)


if __name__ == "__main__":
    main()
