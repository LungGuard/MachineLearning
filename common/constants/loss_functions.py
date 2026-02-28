from enum import StrEnum,Enum


class Loss(StrEnum):
    DEFAULT="loss"
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'

    def get_variant(self, variant):
        variant_prefix = variant.value if isinstance(variant, Enum) else variant
        return f"{variant_prefix}{self.value}" if "_" in variant_prefix else f"{variant_prefix}_{self.value}"
