
from enum import StrEnum,Enum

class Metrics(StrEnum):
    METRIC_ACCURACY = 'accuracy'
    METRIC_PRECISION = 'precision'
    METRIC_RECALL = 'recall'
    METRIC_AUC = 'auc'
    DEFAULT_METRIC_LOSS='loss'
    METRIC_R2='r2'
    METRIC_RMSE=''

    def get_variant(self,variant):
        
        variant_prefix = variant.value if isinstance(variant,Enum) else variant
            
        return f"{variant_prefix}{self.value}" if variant_prefix.contains("_") else f"{variant_prefix}_{self.value}"