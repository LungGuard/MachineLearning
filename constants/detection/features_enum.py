from enum import StrEnum
class Features(StrEnum):
        DIAMETER_MM = 'diameter_mm'
        MALIGNANCY = 'malignancy'
        SPICULATION = 'spiculation'
        LOBULATION = 'lobulation'
        SUBTLETY = 'subtlety'
        SPHERICITY = 'sphericity'
        MARGIN='margin'
        TEXTURE = 'texture'
        CALCIFICATION = 'calcification'
        INTERNAL_STRUCTURE = 'internal_structure'
        ANNOTATION_COUNT = 'annotation_count'

DEFAULT_FEATURES = {
            Features.MALIGNANCY.value : 3.0,  # Indeterminate
            Features.SPICULATION.value : 1.0,
            Features.LOBULATION.value : 1.0,
            Features.SUBTLETY.value : 3.0,
            Features.SPHERICITY.value : 3.0,
            Features.MARGIN.value: 3.0,
            Features.TEXTURE.value: 3.0,
            Features.CALCIFICATION.value : 1.0,
            Features.INTERNAL_STRUCTURE.value: 1.0,
                   }
