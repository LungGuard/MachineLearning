from enum import StrEnum


class ModelStage(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def prefix(self):
        return f"{self.value}_"
