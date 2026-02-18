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
                
