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
        logger.info("Using CPU for all operations")

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
        
        nifti_files = self._convert_dicom_files(series_dirs, output_dir)
        common_mask = self._create_common_uterus_mask(nifti_files, output_dir)
        hemorrhage_results = self._detect_hemorrhage_by_phase(nifti_files, common_mask, output_dir)
        self._create_overlay_visualizations(nifti_files, hemorrhage_results, common_mask, output_dir)

        result = {
            "patient_name": patient_dir.name,
            "processing_time": time.time() - start_time,
            "uterus_mask_volume": self._calculate_volume(common_mask),
        }
        
        # Add per-phase volumes
        for phase, phase_result in hemorrhage_results.items():
            result[f"hemorrhage_volume_{phase}"] = phase_result["volume_ml"]
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Analysis completed in {result['processing_time']:.1f}s")
        return result

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
        if common_mask_path.exists():
            return sitk.ReadImage(str(common_mask_path))
        # Create individual masks (keep in memory, don't write intermediate files)
        individual_masks = {}
        for phase, nifti_path in nifti_files.items():
            reference_img = sitk.ReadImage(str(nifti_path))
            mask = self._create_individual_mask(reference_img, nifti_path, output_dir, phase)
            mask = self._apply_morphology(mask)
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

        common_mask = self._keep_largest_components(common_mask)
        sitk.WriteImage(common_mask, str(common_mask_path))
        return common_mask
    
    def _create_individual_mask(self, reference_img: sitk.Image, nifti_path: Path, output_dir: Path, phase: str) -> sitk.Image:
        """Create individual uterus mask by exclusion"""
        ref_array = sitk.GetArrayFromImage(reference_img)
        
        # Initialize mask array
        mask_array = np.ones(ref_array.shape, dtype=np.uint8)
        
        # Process segmentations
        for task in ["total", "tissue_types", "vertebrae_body"]:
            seg_file = output_dir / f"{task}_{phase}.nii"
            if not seg_file.exists():
                self._run_totalsegmentator(nifti_path, seg_file, task)
            
            seg_img = sitk.ReadImage(str(seg_file))
            if not self._spatial_match(seg_img, reference_img):
                seg_img = self._resample_image(seg_img, reference_img)

            seg_array = sitk.GetArrayFromImage(seg_img)
            seg_array = (seg_array > 0) & (seg_array != 21)
            seg_img = sitk.GetImageFromArray(seg_array.astype(np.uint8))
            if task in ["tissue_types", "vertebrae_body"]:
                seg_img = sitk.BinaryDilate(seg_img, [self.config.morphology_radius] * 3, sitk.sitkBall, 0, 1)
            seg_array = sitk.GetArrayFromImage(seg_img)
            mask_array[seg_array > 0] = 0
        
        # Apply HU filtering
        mask_array = self._apply_hu_filtering(mask_array, ref_array, phase)
        
        # Convert back to SimpleITK
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(reference_img)
        return mask
    
    def _apply_hu_filtering(self, mask_array, ref_data, phase):
        min_hu, max_hu = getattr(self.config, f'{phase}_hu')
        
        mask_array[(ref_data < min_hu) | (ref_data > max_hu)] = 0
        return mask_array

    
    def _run_totalsegmentator(self, input_path: Path, output_file: Path, task: str):
        """Run TotalSegmentator with specified task"""
        cmd = ["TotalSegmentator", "-i", str(input_path), "-o", str(output_file), "--task", task, "--ml"]
        
        if self.config.use_fast_mode:
            cmd.append("--fast")
        if self.config.device != "auto":
            cmd.extend(["--device", self.config.device])
        
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)

    def _detect_hemorrhage_by_phase(self, nifti_files: Dict[str, Path], common_uterus_mask: sitk.Image, output_dir: Path) -> Dict[str, Dict]:
        """Detect hemorrhage in each available phase separately"""
        results = {}
        
        for phase, nifti_path in nifti_files.items():
            logger.info(f"Processing {phase} phase for hemorrhage detection")
            
            # Load and smooth image
            image = sitk.ReadImage(str(nifti_path))

            hemorrhage_mask = sitk.BinaryThreshold(image, getattr(self.config, f"{phase}_threshold"), 500, 1, 0)
            hemorrhage_mask = sitk.BinaryMorphologicalOpening(hemorrhage_mask, [1] * 3, sitk.sitkBall)

            # Ensure spatial alignment with uterus mask
            if not self._spatial_match(hemorrhage_mask, common_uterus_mask):
                hemorrhage_mask = self._resample_image(hemorrhage_mask, common_uterus_mask)
            
            # Apply uterus mask
            hemorrhage_mask = sitk.Multiply(hemorrhage_mask, common_uterus_mask)
            
            # Calculate volume
            volume_ml = self._calculate_volume(hemorrhage_mask)
            
            # Save mask
            mask_path = output_dir / f"hemorrhage_mask_{phase}.nii.gz"
            sitk.WriteImage(hemorrhage_mask, str(mask_path))
            
            results[phase] = {
                "mask": hemorrhage_mask,
                "volume_ml": volume_ml,
                "mask_path": mask_path
            }
            
            logger.info(f"{phase.capitalize()} hemorrhage volume: {volume_ml:.2f} mL")
        
        arterial = sitk.ReadImage(str(nifti_files["arterial"]))
        portal = sitk.ReadImage(str(nifti_files["portal"]))
        if not self._spatial_match(arterial, portal):
            arterial = self._resample_image(arterial, portal)
        portal = sitk.Subtract(portal, arterial)
        portal = sitk.BinaryThreshold(portal, self.config.subtract, 500, 1, 0)
        portal = sitk.BinaryMorphologicalOpening(portal, [1] * 3, sitk.sitkBall)
        if not self._spatial_match(portal, common_uterus_mask):
            portal = self._resample_image(portal, common_uterus_mask)
        portal = sitk.Multiply(portal, common_uterus_mask)
        results["subtract"] = {
            "mask": portal,
            "volume_ml": self._calculate_volume(portal),
            "mask_path": output_dir / "hemorrhage_mask_subtract.nii.gz"
        }
        sitk.WriteImage(portal, str(results["subtract"]["mask_path"]))

        return results

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
        radius = self.config.morphology_radius
        
        mask = sitk.BinaryMorphologicalOpening(mask, [radius] * 3, sitk.sitkBall)
        mask = sitk.BinaryMorphologicalClosing(mask, [radius] * 3, sitk.sitkBall)
        hole_fill = sitk.BinaryFillholeImageFilter()
        return hole_fill.Execute(mask)
    
    def _calculate_volume(self, mask: sitk.Image) -> float:
        """Calculate volume in mL - optimized version"""
        spacing = mask.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        
        # Use numpy for faster counting
        mask_array = sitk.GetArrayFromImage(mask)
        total_voxels = np.count_nonzero(mask_array)
        
        return (total_voxels * voxel_volume_mm3) / 1000.0

    def _create_overlay_visualizations(self, nifti_files: dict,
                                        hemorrhage_results: dict,
                                        common_uterus_mask: sitk.Image,
                                        output_dir: Path):
        """Create overlay visualizations"""
        nifti_files["subtract"] = nifti_files["portal"]
        for phase, nifti_file in nifti_files.items():
            hemorrhage_mask = hemorrhage_results[phase]["mask"]
            original_img = sitk.ReadImage(str(nifti_file))

            # Ensure spatial alignment
            if not self._spatial_match(hemorrhage_mask, original_img):
                hemorrhage_mask = self._resample_image(hemorrhage_mask, original_img)
            
            aligned_common_mask = common_uterus_mask
            if not self._spatial_match(aligned_common_mask, original_img):
                aligned_common_mask = self._resample_image(aligned_common_mask, original_img)
                
            # Create overlay
            original_normalized = self._normalize_image_for_display(original_img)
            overlay_array = sitk.GetArrayFromImage(original_normalized)

            # Uterus contour
            uterus_contour_img = sitk.BinaryContour(aligned_common_mask, fullyConnected=True)
            uterus_contour_array = sitk.GetArrayViewFromImage(uterus_contour_img)
            overlay_array[uterus_contour_array > 0] = 255

            # Hemorrhage contour
            hemorrhage_contour_img = sitk.BinaryContour(hemorrhage_mask, fullyConnected=True)
            hemorrhage_contour_array = sitk.GetArrayViewFromImage(hemorrhage_contour_img)
            overlay_array[hemorrhage_contour_array > 0] = 0
            
            overlay_img = sitk.GetImageFromArray(overlay_array)
            overlay_img.CopyInformation(original_img)
            overlay_output_path = output_dir / f"{phase}_overlay.nii.gz"
            sitk.WriteImage(overlay_img, str(overlay_output_path))
    
    def _normalize_image_for_display(self, image: sitk.Image) -> sitk.Image:
        """Normalize image to 0-255 range for display"""
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        image_windowed = sitk.Clamp(image_float, sitk.sitkFloat32, -160, 240)
        image_normalized = sitk.RescaleIntensity(image_windowed, 0, 255)
        return sitk.Cast(image_normalized, sitk.sitkUInt8)

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
        mask_array = sitk.GetArrayFromImage(mask)
        if np.sum(mask_array) == 0:
            return mask
        
        # Use SimpleITK for connected components analysis
        labeled_img = sitk.ConnectedComponent(mask)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(labeled_img)
        
        labels = label_stats.GetLabels()
        if len(labels) <= 2:
            return mask
        
        # Keep top 2 largest components
        label_sizes = [(label, label_stats.GetNumberOfPixels(label)) for label in labels]
        label_sizes.sort(key=lambda x: x[1], reverse=True)
        keep_labels = [label for label, _ in label_sizes[:1]]
        
        # CPU mask creation
        labeled_array = sitk.GetArrayFromImage(labeled_img)
        new_mask_array = np.zeros_like(mask_array, dtype=np.uint8)
        for label in keep_labels:
            new_mask_array[labeled_array == label] = 1
        
        new_mask = sitk.GetImageFromArray(new_mask_array)
        new_mask.CopyInformation(mask)
        return new_mask