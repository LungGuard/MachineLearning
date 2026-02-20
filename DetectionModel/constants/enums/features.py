from enum import StrEnum


class Features(StrEnum):
    DIAMETER_MM = 'diameter_mm'
    MALIGNANCY = 'malignancy'
    SPICULATION = 'spiculation'
    LOBULATION = 'lobulation'
    SUBTLETY = 'subtlety'
    SPHERICITY = 'sphericity'
    MARGIN = 'margin'
    TEXTURE = 'texture'
    CALCIFICATION = 'calcification'
    INTERNAL_STRUCTURE = 'internal_structure'
    ANNOTATION_COUNT = 'annotation_count'


DEFAULT_FEATURES = {
    Features.MALIGNANCY: 3.0,
    Features.SPICULATION: 1.0,
    Features.LOBULATION: 1.0,
    Features.SUBTLETY: 3.0,
    Features.SPHERICITY: 3.0,
    Features.MARGIN: 3.0,
    Features.TEXTURE: 3.0,
    Features.CALCIFICATION: 1.0,
    Features.INTERNAL_STRUCTURE: 1.0,
}
