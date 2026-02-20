from enum import StrEnum, Enum


class Metrics(StrEnum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    AUC = 'auc'
    DEFAULT_LOSS = 'loss'
    R2 = 'r2'
    RMSE = 'rmse'
    MAE = "mae"

    def get_variant(self, variant):
        variant_prefix = variant.value if isinstance(variant, Enum) else variant
        return f"{variant_prefix}{self.value}" if "_" in variant_prefix else f"{variant_prefix}_{self.value}"
