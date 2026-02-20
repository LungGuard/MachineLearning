from enum import IntEnum

class PreProcessingConstants:


    

    TARGET_SPACING = (1.0, 1.0, 1.0)
    WINDOW_CENTER = -600.0
    WINDOW_WIDTH = 1500.0

    MAX_CROP_SCALE = 1.5




class HUValues(IntEnum):
        AIR_HU = -1000
        MAX_HU = 3000
        PADDING_THRESHOLD = -1500
        OFFSET_THRESHOLD = -100
        OFFSET_CORRECTION = 1024
        VALID_PIXEL_MAX = 4000
        OFFSET_PERCENTILE = 5


class IntensityRange(IntEnum):
        OUTPUT_MIN = 0.0
        OUTPUT_MAX = 1.0