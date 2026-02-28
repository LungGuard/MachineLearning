from enum import StrEnum,Enum
from ..mixins import EnumMixin

class Loss(StrEnum,EnumMixin):
    DEFAULT="loss"
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    