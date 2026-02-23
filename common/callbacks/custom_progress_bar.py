import sys

import torch
from lightning.pytorch.callbacks import Callback

# Bypass Rich's FileProxy that wraps sys.stdout in Jupyter/Lightning
_STDOUT = sys.__stdout__


def _print(*args, **kwargs):
    print(*args, file=_STDOUT, **kwargs)


class CustomProgressBar(Callback):
    def __init__(self, comparator=None, mapper=None, refresh_rate=10, bar_length=30):
        super().__init__()
        self.comparator = comparator
        self.mapper = mapper
        self.refresh_rate = refresh_rate
        self.bar_length = bar_length

    def get_metrics(self, trainer):
        items = trainer.callback_metrics.copy()
        items.pop("v_num", None)

        items = dict(sorted(items.items(), key=self.comparator)) if self.comparator else items
        items = dict(map(self.mapper, items.items())) if self.mapper else items

        return items

    def _format_value(self, key, value):
        formatter_map = {
            float: lambda k, v: f"{k}: {v:.4f}",
            torch.Tensor: lambda k, v: f"{k}: {v.item():.4f}",
        }

        matched_formatter = next(
            (
                fmt
                for type_key, fmt in formatter_map.items()
                if isinstance(value, type_key) and not (type_key == torch.Tensor and value.numel() != 1)
            ),
            lambda k, v: f"{k}: {v}",
        )

        return matched_formatter(key, value)

    def _format_metrics(self, metrics_dict):
        return " | ".join(
            self._format_value(k, v) for k, v in metrics_dict.items()
        )

    def _render_bar(self, current, total, prefix="", metrics_str=""):
        percent = current / max(total, 1)
        filled = int(self.bar_length * percent)
        bar = '█' * filled + '-' * (self.bar_length - filled)

        _STDOUT.write(f"\r{prefix} [{bar}] {int(percent * 100)}% | {metrics_str}\033[K")
        _STDOUT.flush()

    def _should_refresh(self, batch_idx, total_batches):
        return batch_idx % self.refresh_rate == 0 or batch_idx == total_batches - 1

    def _get_val_total(self, trainer, dataloader_idx):
        batches_list = (
            trainer.num_val_batches
            if isinstance(trainer.num_val_batches, list)
            else [trainer.num_val_batches]
        )

        return batches_list[dataloader_idx]

    # ── Training Hooks ──────────────────────────────────────────────

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        _print(f"\n{'=' * 80}")
        _print(f"🚀 EPOCH {epoch}/{max_epochs}")
        _print(f"{'=' * 80}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._render_bar(
            current=batch_idx + 1,
            total=trainer.num_training_batches,
            prefix="🏋️  Training   ",
            metrics_str=self._format_metrics(self.get_metrics(trainer)),
        ) if self._should_refresh(batch_idx, trainer.num_training_batches) else None

    def on_train_epoch_end(self, trainer, pl_module):
        _print()

    # ── Validation Hooks ────────────────────────────────────────────

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        (
            self._render_bar(
                current=batch_idx + 1,
                total=self._get_val_total(trainer, dataloader_idx),
                prefix="🧪 Validation ",
                metrics_str=self._format_metrics(self.get_metrics(trainer)),
            )
            if self._should_refresh(batch_idx, self._get_val_total(trainer, dataloader_idx))
            else None
        ) if not trainer.sanity_checking else None

    def on_validation_epoch_end(self, trainer, pl_module):
        _print()
        metrics_str = self._format_metrics(self.get_metrics(trainer))
        _print(f"📊 Epoch {trainer.current_epoch + 1} Summary:")
        _print(f"   {metrics_str}")
        _print(f"{'-' * 80}")