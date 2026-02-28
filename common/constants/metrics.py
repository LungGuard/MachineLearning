from enum import StrEnum, Enum
from ..mixins import EnumMixin


class Metrics(StrEnum,EnumMixin):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    AUC = 'auc'
    DEFAULT_LOSS = 'loss'
    R2 = 'r2'
    RMSE = 'rmse'
    MAE = "mae"