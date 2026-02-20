from dataclasses import dataclass


@dataclass
class TransformValues:
    horizontal_flip_probability: float = 0.5
    rotate_angle_range: float = 10
    brightness_factor: float = 0.1
    contrast_factor: float = 0.1
