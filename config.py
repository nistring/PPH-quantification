from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class Config:
    """Simplified configuration for PPH analysis"""
    
    # HU thresholds for hemorrhage detection (min_hu, max_hu)
    arterial_hu: Tuple[int, int] = (50, 100)      # No lower limit, upper limit 150 HU
    portal_hu: Tuple[int, int] = (50, 100)        # No lower limit, upper limit 100 HU
    arterial_th: float = 100.0  # Threshold for arterial phase detection
    portal_th: float = 50.0    # Threshold for portal phase detection

    # Processing parameters
    min_component_size: int = 50
    morphology_radius: int = 5
    
    # TotalSegmentator settings
    use_fast_mode: bool = False
    device: str = "gpu:0"
    
    # Output settings
    temp_dir: str = "temp"
    output_dir: str = "output"
    
    # Mask refinement parameters
    max_components: int = 1           # Maximum connected components to keep
    region_growing_sigma: float = 1.0 # Gaussian smoothing sigma for boundary smoothing
    
    # Debug settings
    save_debug_masks: bool = False    # Save intermediate masks for debugging
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)