import logging
import time
import json
from pathlib import Path
from typing import Dict, Any

import SimpleITK as sitk
import numpy as np
import subprocess

from config import Config
from dicom_processor import DICOMProcessor

logger = logging.getLogger(__name__)

class PPHAnalyzer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dicom_processor = DICOMProcessor(self.config.temp_dir)
        
    def analyze_patient(self, patient_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Analyze patient with available phases"""
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing patient: {patient_dir.name}")
        
        # Find available phases
        series_dirs = {}
        for phase, subdir_name in [("arterial", "Arterial"), ("portal", "Portal")]:
            series_dir = patient_dir / subdir_name
            if series_dir.exists() and any(series_dir.iterdir()):
                series_dirs[phase] = series_dir
                logger.info(f"Found {phase} phase")
        
        if not series_dirs:
            raise FileNotFoundError(f"No valid series found in {patient_dir}")
        
        try:
            nifti_files = self._convert_dicom_files(series_dirs, output_dir)
            common_mask = self._create_common_uterus_mask(nifti_files, output_dir)
            hemorrhage_results, hemorrhage_masks = self._detect_hemorrhage(nifti_files, common_mask, output_dir)
            self._create_overlay_visualizations(nifti_files, common_mask, hemorrhage_masks, output_dir)
            
            results = {
                "patient_name": patient_dir.name,
                "processing_time": time.time() - start_time,
                "hemorrhage_volumes": {phase: data["volume_ml"] for phase, data in hemorrhage_results.items()},
                "uterus_mask_volume": self._calculate_volume(common_mask),
            }
            
            with open(output_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Analysis completed in {results['processing_time']:.1f}s")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _convert_dicom_files(self, series_dirs: Dict[str, Path], output_dir: Path) -> Dict[str, Path]:
        """Convert DICOM files to NIfTI format"""
        nifti_files = {}
        for phase, series_dir in series_dirs.items():
            nifti_path = output_dir / f"{phase}.nii.gz"
            if not nifti_path.exists():
                self.dicom_processor.dicom_to_nifti(series_dir, nifti_path)
            nifti_files[phase] = nifti_path
        return nifti_files
    
    def _create_common_uterus_mask(self, nifti_files: Dict[str, Path], output_dir: Path) -> sitk.Image:
        """Create common uterus mask"""
        common_mask_path = output_dir / "common_uterus_mask.nii.gz"
        
        # Create individual masks
        individual_masks = {}
        for phase, nifti_path in nifti_files.items():
            mask_path = output_dir / f"uterus_mask_{phase}.nii.gz"
            mask = self._create_individual_mask(nifti_path, output_dir, phase)
            mask = self._apply_morphology(mask)
            sitk.WriteImage(mask, str(mask_path))
            individual_masks[phase] = mask
        
        # Create common mask
        phases = list(individual_masks.keys())
        common_mask = individual_masks[phases[0]]
        
        if len(phases) > 1:
            for i in range(1, len(phases)):
                other_mask = individual_masks[phases[i]]
                if not self._spatial_match(other_mask, common_mask):
                    other_mask = self._resample_image(other_mask, common_mask)
                common_mask = sitk.Multiply(common_mask, other_mask)
            logger.info(f"Created intersection mask from phases: {phases}")
        else:
            logger.info(f"Using single phase ({phases[0]}) mask")
        
        sitk.WriteImage(common_mask, str(common_mask_path))
        return common_mask
    
    def _create_individual_mask(self, nifti_path: Path, output_dir: Path, phase: str) -> sitk.Image:
        """Create individual uterus mask by exclusion"""
        reference_img = sitk.ReadImage(str(nifti_path))
        uterus_array = np.ones(sitk.GetArrayFromImage(reference_img).shape, dtype=np.uint8)
        
        # Run TotalSegmentator tasks and apply exclusions
        tasks = ["total", "tissue_types", "vertebrae_body"]
        for task in tasks:
            seg_file = output_dir / f"{task}_{phase}.nii"
            if not seg_file.exists():
                self._run_totalsegmentator(nifti_path, seg_file, task)
            
            seg_img = sitk.ReadImage(str(seg_file))
            if seg_img.GetSize() != reference_img.GetSize():
                seg_img = self._resample_image(seg_img, reference_img)
            
            if task in ["tissue_types", "vertebrae_body"]:
                seg_img = sitk.BinaryDilate(seg_img, [self.config.morphology_radius] * 3, sitk.sitkBall, 0, 1)
            
            seg_array = sitk.GetArrayFromImage(seg_img)
            uterus_array[(seg_array > 0) & (seg_array != 21)] = 0  # Exclude bladder

        # Apply HU filtering
        ref_array = sitk.GetArrayFromImage(reference_img)
        uterus_array = self._apply_filtering(uterus_array, ref_array, phase)
        uterus_mask = sitk.GetImageFromArray(uterus_array)
        uterus_mask.CopyInformation(reference_img)
        
        # Keep largest components
        return self._keep_largest_components(uterus_mask)
    
    def _run_totalsegmentator(self, input_path: Path, output_file: Path, task: str):
        """Run TotalSegmentator with specified task"""
        cmd = ["TotalSegmentator", "-i", str(input_path), "-o", str(output_file), "--task", task, "--ml"]
        
        if self.config.use_fast_mode:
            cmd.append("--fast")
        if self.config.device != "auto":
            cmd.extend(["--device", self.config.device])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"TotalSegmentator {task} task failed")

    def _detect_hemorrhage(self, nifti_files: Dict[str, Path], common_uterus_mask: sitk.Image, output_dir: Path) -> tuple[Dict[str, Dict], Dict[str, sitk.Image]]:
        """Detect hemorrhage in available phases"""
        results = {}
        hemorrhage_masks = {}
        
        thresholds = {"arterial": getattr(self.config, 'arterial_th', 150), "portal": getattr(self.config, 'portal_th', 120)}
        
        for phase, nifti_path in nifti_files.items():
            hu_threshold = thresholds.get(phase, thresholds["arterial"])
            
            hemorrhage_mask = sitk.BinaryThreshold(sitk.ReadImage(str(nifti_path)), hu_threshold, 1000, 1, 0)
            if hemorrhage_mask.GetSize() != common_uterus_mask.GetSize():
                hemorrhage_mask = self._resample_image(hemorrhage_mask, common_uterus_mask)
            hemorrhage_mask = sitk.Multiply(hemorrhage_mask, common_uterus_mask)
            
            hemorrhage_mask_path = output_dir / f"{phase}_hemorrhage.nii.gz"
            sitk.WriteImage(hemorrhage_mask, str(hemorrhage_mask_path))
            
            hemorrhage_masks[phase] = hemorrhage_mask
            results[phase] = {
                "volume_ml": self._calculate_volume(hemorrhage_mask),
                "hu_threshold": hu_threshold,
                "mask_file": hemorrhage_mask_path.name
            }
            
            logger.info(f"Detected hemorrhage in {phase} phase: {results[phase]['volume_ml']:.2f} mL")
        
        return results, hemorrhage_masks
    
    def _resample_image(self, image: sitk.Image, reference: sitk.Image) -> sitk.Image:
        """Resample image to match reference with proper spatial alignment"""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        
        # Set output parameters explicitly to match reference
        resampler.SetSize(reference.GetSize())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        
        return resampler.Execute(image)
    
    def _apply_morphology(self, mask: sitk.Image) -> sitk.Image:
        """Apply morphological operations"""
        hole_fill = sitk.VotingBinaryIterativeHoleFillingImageFilter()
        hole_fill.SetRadius([self.config.morphology_radius] * 3)
        mask = hole_fill.Execute(mask)
        
        mask = sitk.BinaryMorphologicalClosing(mask, [self.config.morphology_radius] * 3)
        mask = sitk.BinaryMorphologicalOpening(mask, [2] * 3)
        
        return mask

    def _apply_filtering(self, uterus_array: np.ndarray, ref_array: np.ndarray, phase: str) -> np.ndarray:
        """Apply HU filtering with fallback values"""
        hu_ranges = {"arterial": (-50, 200), "portal": (-50, 180)}
        
        if phase == "arterial" and hasattr(self.config, 'arterial_hu'):
            min_hu, max_hu = self.config.arterial_hu
        elif phase == "portal" and hasattr(self.config, 'portal_hu'):
            min_hu, max_hu = self.config.portal_hu
        else:
            min_hu, max_hu = hu_ranges.get(phase, hu_ranges["arterial"])
            logger.warning(f"Using default HU range for {phase} phase: {min_hu} to {max_hu}")
        
        uterus_array[(ref_array < min_hu) | (ref_array > max_hu)] = 0
        return uterus_array
    
    def _calculate_volume(self, mask: sitk.Image) -> float:
        """Calculate volume in mL"""
        mask_array = sitk.GetArrayFromImage(mask)
        spacing = mask.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        total_voxels = np.sum(mask_array > 0)
        return (total_voxels * voxel_volume_mm3) / 1000.0

    def _create_overlay_visualizations(self, nifti_files: Dict[str, Path], common_uterus_mask: sitk.Image,
                                        hemorrhage_masks: Dict[str, sitk.Image], output_dir: Path):
        """Create overlay visualizations"""
        logger.info("Creating overlay visualizations...")
        
        for phase, nifti_path in nifti_files.items():
            original_img = sitk.ReadImage(str(nifti_path))
            phase_hemorrhage_mask = hemorrhage_masks[phase]
            
            # Ensure spatial alignment
            if phase_hemorrhage_mask.GetSize() != original_img.GetSize():
                phase_hemorrhage_mask = self._resample_image(phase_hemorrhage_mask, original_img)
            if common_uterus_mask.GetSize() != original_img.GetSize():
                common_uterus_mask = self._resample_image(common_uterus_mask, original_img)
            
            # Create overlay
            uterus_surface = self._extract_surface(common_uterus_mask)
            uterus_surface_array = sitk.GetArrayFromImage(uterus_surface)
            hemorrhage_array = sitk.GetArrayFromImage(phase_hemorrhage_mask)          
            
            original_normalized = self._normalize_image_for_display(original_img)
            original_normalized = sitk.GetArrayFromImage(original_normalized)
            original_normalized[uterus_surface_array > 0] = 255  # Highlight uterus surface
            original_normalized[hemorrhage_array > 0] = 0  # Highlight hemorrhage
            original_normalized = sitk.GetImageFromArray(original_normalized)
            original_normalized.CopyInformation(original_img)
            
            overlay_output_path = output_dir / f"{phase}_overlay.nii.gz"
            sitk.WriteImage(original_normalized, str(overlay_output_path))
            
            logger.info(f"Created overlay visualization for {phase} phase: {overlay_output_path.name}")
    
    def _normalize_image_for_display(self, image: sitk.Image) -> sitk.Image:
        """Normalize image to 0-255 range for display purposes"""
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        # Use windowing appropriate for CT images (center=40, width=400)
        image_windowed = sitk.Clamp(image_float, sitk.sitkFloat32, -160, 240)
        image_normalized = sitk.RescaleIntensity(image_windowed, 0, 255)
        return sitk.Cast(image_normalized, sitk.sitkUInt8)

    def _extract_surface(self, mask: sitk.Image, thickness: int = 1) -> sitk.Image:
        """Extract the outer surface of a binary mask using morphological gradient"""
        radius = [thickness] * 3
        dilated = sitk.BinaryDilate(mask, radius, sitk.sitkBall, 0, 1)
        return sitk.Subtract(dilated, mask)
    
    def _cleanup(self):
        """Clean up temporary files"""
        try:
            self.dicom_processor.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _spatial_match(self, image1: sitk.Image, image2: sitk.Image) -> bool:
        """Check if two images have matching spatial properties"""
        return (image1.GetSize() == image2.GetSize() and 
                image1.GetSpacing() == image2.GetSpacing() and
                image1.GetOrigin() == image2.GetOrigin())
    
    def _keep_largest_components(self, mask: sitk.Image) -> sitk.Image:
        """Keep largest connected components"""
        labeled_img = sitk.ConnectedComponent(mask)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(labeled_img)
        
        labels = label_stats.GetLabels()
        if not labels:
            return mask

        # Sort by size and keep largest components
        label_sizes = [(label, label_stats.GetNumberOfPixels(label)) for label in labels]
        label_sizes.sort(key=lambda x: x[1], reverse=True)
        keep_labels = [label for label, _ in label_sizes[:self.config.max_components]]

        new_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        new_mask.CopyInformation(mask)

        for label in keep_labels:
            component = sitk.BinaryThreshold(labeled_img, label, label, 1, 0)
            new_mask = sitk.Add(new_mask, component)
        
        return new_mask