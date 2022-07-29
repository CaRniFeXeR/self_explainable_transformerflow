from typing import Dict
from dataclasses import dataclass

@dataclass
class DatashiftAugmentationConfig:
    shift_propability : float
    shift_percent : float
    polygon_scale_range : Dict[str, float]
    scale_propability : float
    scale_propability_2nd_marker : float
    disable_augmentation : bool = False
    enable_shear_augmentation : bool = False
