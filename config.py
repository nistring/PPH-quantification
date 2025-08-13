from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class Config:
    """Simplified configuration for PPH analysis"""
    
    # HU thresholds for hemorrhage detection (min_hu, max_hu)
    arterial_hu: Tuple[int, int] = (0, 250)
    portal_hu: Tuple[int, int] = (0, 250)
    arterial_threshold: float = 160  # Threshold
    portal_threshold: float = 180  # Threshold
    subtract: int = 130  # Subtraction threshold

    # Processing parameters
    morphology_radius: int = 3
    
    # TotalSegmentator settings
    use_fast_mode: bool = False
    device: str = "gpu:0"

    # Output settings
    temp_dir: str = "temp"
    output_dir: str = "output"
    
    # Mask refinement parameters
    max_components: int = 1           # Maximum connected components to keep
    
    # Debug settings
    save_debug_masks: bool = False    # Save intermediate masks for debugging
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)