from enum import StrEnum,auto


class CancerType(StrEnum):
    ADENOCARCINOMA = auto()
    LARGE_CELL_CARCINOMA = auto()
    SQUAMOUS_CELL_CARCINOMA = auto()
    NORMAL = auto()