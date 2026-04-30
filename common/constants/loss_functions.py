from enum import StrEnum,Enum
from ..mixins import EnumMixin

class Loss(EnumMixin, StrEnum):
    DEFAULT="loss"
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    