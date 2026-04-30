from enum import StrEnum, Enum
from ..mixins import EnumMixin


class Metrics(EnumMixin, StrEnum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    AUC = 'auc'
    R2 = 'r2'
    RMSE = 'rmse'
    MAE = "mae"