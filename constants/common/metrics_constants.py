
from enum import StrEnum,Enum

class Metrics(StrEnum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    AUC = 'auc'
    DEFAULT_LOSS='loss'
    R2='r2'
    RMSE='rmse'

    def get_variant(self,variant):
        
        variant_prefix = variant.value if isinstance(variant,Enum) else variant
            
        return f"{variant_prefix}{self.value}" if variant_prefix.contains("_") else f"{variant_prefix}_{self.value}"