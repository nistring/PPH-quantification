from dataclasses import dataclass
from typing import Tuple, List
import os

@dataclass
class Config:
    """Simplified configuration for PPH analysis"""
    
    # HU thresholds for hemorrhage detection
    non_enhanced_hu: Tuple[int, int] = (50, 75)
    arterial_hu: Tuple[int, int] = (85, 370)
    portal_hu: Tuple[int, int] = (85, 370)
    
    # Processing parameters
    min_component_size: int = 50
    morphology_radius: int = 5
    
    # TotalSegmentator settings
    use_fast_mode: bool = False
    device: str = "gpu:0"
    
    # Output settings
    temp_dir: str = "temp"
    output_dir: str = "output"
    
    # Clinical thresholds (mL)
    mild_threshold: float = 50.0
    moderate_threshold: float = 200.0
    severe_threshold: float = 500.0
    critical_threshold: float = 1000.0
    
    # Mask refinement parameters
    min_uterus_hu: int = 20           # Minimum HU for uterine tissue
    max_uterus_hu: int = 1000          # Maximum HU for uterine tissue  
    max_components: int = 1           # Maximum connected components to keep
    x_margin: float = (0.3, 0.7)
    y_margin: float = (0.45, 0.75)
    z_margin: float = (0.3, 0.7)
    region_growing_sigma: float = 1.0 # Gaussian smoothing sigma for boundary smoothing
    
    # Debug settings
    save_debug_masks: bool = False    # Save intermediate masks for debugging
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_severity(self, volume_ml: float) -> str:
        """Get severity classification"""
        if volume_ml >= self.critical_threshold:
            return "Critical"
        elif volume_ml >= self.severe_threshold:
            return "Severe"
        elif volume_ml >= self.moderate_threshold:
            return "Moderate"
        elif volume_ml >= self.mild_threshold:
            return "Mild"
        else:
            return "Minimal"
