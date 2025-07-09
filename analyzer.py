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

class SimplePPHAnalyzer:
    """Simplified three-phase PPH analyzer"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dicom_processor = DICOMProcessor(self.config.temp_dir)
        
    def analyze_patient(self, patient_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Analyze patient with three phases
        
        Args:
            patient_dir: Directory with 201/, 601/, 701/ subdirectories
            output_dir: Output directory
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing patient: {patient_dir.name}")
        
        results_file = output_dir / "results.json"
        
        # Check required series exist
        series_dirs = {
            "non_enhanced": patient_dir / "201",
            "arterial": patient_dir / "601", 
            "portal": patient_dir / "701"
        }
        
        for phase, series_dir in series_dirs.items():
            if not series_dir.exists():
                raise FileNotFoundError(f"Missing series {series_dir}")
        
        try:
            # Step 1: Convert DICOM to NIfTI (check if already exists)
            logger.info("Converting DICOM to NIfTI...")
            nifti_files = {}
            for phase, series_dir in series_dirs.items():
                nifti_path = output_dir / f"{phase}.nii.gz"
                if nifti_path.exists():
                    logger.info(f"NIfTI file already exists for {phase}, skipping conversion...")
                    nifti_files[phase] = nifti_path
                else:
                    self.dicom_processor.dicom_to_nifti(series_dir, nifti_path)
                    nifti_files[phase] = nifti_path
            
            # Step 2: Create uterus masks for each phase using TotalSegmentator exclusion
            logger.info("Creating uterus masks for each phase...")
            uterus_masks = self._create_uterus_masks_all_phases(nifti_files, output_dir)
            
            # Step 3: Detect hemorrhage in each phase using respective uterus masks
            logger.info("Detecting hemorrhage...")
            hemorrhage_results = self._detect_hemorrhage_all_phases(
                nifti_files, uterus_masks, output_dir
            )
            
            # Step 4: Analyze results
            analysis = self._analyze_results(hemorrhage_results)
            
            # Compile final results
            results = {
                "patient_name": patient_dir.name,
                "processing_time": time.time() - start_time,
                "hemorrhage_volumes": {
                    phase: data["volume_ml"] for phase, data in hemorrhage_results.items()
                },
                "uterus_mask_volumes": {
                    phase: self._calculate_volume(mask) for phase, mask in uterus_masks.items()
                },
                "analysis": analysis,
                "output_dir": str(output_dir)
            }
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Analysis completed in {results['processing_time']:.1f}s")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _create_uterus_masks_all_phases(self, nifti_files: Dict[str, Path], output_dir: Path) -> Dict[str, sitk.Image]:
        """Create uterus masks for all three phases using TotalSegmentator exclusion"""
        uterus_masks = {}
        
        for phase, nifti_path in nifti_files.items():
            logger.info(f"Creating uterus mask for {phase} phase...")
            
            # Check if uterus mask already exists
            mask_path = output_dir / f"uterus_mask_{phase}.nii.gz"
            if mask_path.exists():
                logger.info(f"Uterus mask already exists for {phase}, loading existing mask...")
                uterus_mask = sitk.ReadImage(str(mask_path))
                uterus_masks[phase] = uterus_mask
                volume_ml = self._calculate_volume(uterus_mask)
                logger.info(f"Loaded existing uterus mask {phase}: {volume_ml:.2f} mL")
                continue

            # Define output file paths
            total_file = output_dir / f"total_{phase}.nii"
            tissue_types_file = output_dir / f"tissue_types_{phase}.nii"
            vertebrae_body_file = output_dir / f"vertebrae_body_{phase}.nii"
            self._create_uterus_mask_by_exclusion(nifti_path, [total_file, tissue_types_file, vertebrae_body_file], phase)
                
            # Load reference image and create mask from existing segmentation
            reference_img = sitk.ReadImage(str(nifti_path))
            uterus_mask = self._exclude_segmented_files([total_file, tissue_types_file, vertebrae_body_file], reference_img)
            
            # Save uterus mask
            sitk.WriteImage(uterus_mask, str(mask_path))
            
            # Save debug masks if enabled
            if hasattr(self.config, 'save_debug_masks') and self.config.save_debug_masks:
                debug_dir = output_dir / "debug_masks"
                debug_dir.mkdir(exist_ok=True)
                sitk.WriteImage(uterus_mask, str(debug_dir / f"refined_uterus_mask_{phase}.nii.gz"))
            
            uterus_masks[phase] = uterus_mask
            
            # Log mask statistics
            volume_ml = self._calculate_volume(uterus_mask)
            logger.info(f"Uterus mask {phase}: {volume_ml:.2f} mL")
        
        return uterus_masks
    
    def _create_uterus_mask_by_exclusion(self, nifti_path: Path, output_paths: Path, phase: str) -> None:
        """Create uterus mask by excluding all segmented areas from TotalSegmentator"""
        for output_file in output_paths:
            if not output_file.exists():
                self._run_totalsegmentator(nifti_path, output_file, output_file.name.replace(f"_{phase}.nii", ""))
            else:
                logger.info(f"Output file {output_file.name} already exists, skipping TotalSegmentator for {phase} phase...")
    
    def _run_totalsegmentator(self, input_path: Path, output_file: Path, task: str):
        """Run TotalSegmentator with specified task"""
        cmd = [
            "TotalSegmentator",
            "-i", str(input_path),
            "-o", str(output_file),
            "--task", task,
            "--ml",
        ]
        
        if self.config.use_fast_mode:
            cmd.append("--fast")
        
        if self.config.device != "auto":
            cmd.extend(["--device", self.config.device])
        
        try:
            logger.info(f"Running TotalSegmentator {task} task...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
            logger.info(f"TotalSegmentator {task} task completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"TotalSegmentator {task} task failed: {e.stderr}")
            raise RuntimeError(f"TotalSegmentator {task} task failed")
        except subprocess.TimeoutExpired:
            logger.error(f"TotalSegmentator {task} task timed out")
            raise RuntimeError(f"TotalSegmentator {task} task timed out")
    
    def _exclude_segmented_files(self, segmentation_files: list, reference_img: sitk.Image) -> sitk.Image:
        """Create uterus mask by excluding all segmented areas from files"""
        
        # Start with an array of ones (all voxels are candidate uterus)
        mask_shape = sitk.GetArrayFromImage(reference_img).shape
        uterus_array = np.ones(mask_shape, dtype=np.uint8)
        
        # Exclude all segmented areas from the files
        for seg_file in segmentation_files:
            seg_file = Path(seg_file)
            if not seg_file.exists():
                logger.warning(f"Segmentation file {seg_file} does not exist, skipping...")
                continue
                
            logger.info(f"Excluding segmentations from {seg_file.name}")
            
            try:
                # Load segmentation mask
                seg_img = sitk.ReadImage(str(seg_file))
                
                # Resample to match reference image if needed
                if seg_img.GetSize() != reference_img.GetSize():
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(reference_img)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    seg_img = resampler.Execute(seg_img)
                
                if "tissue_types" or "vertebrae_body" in seg_file.name:
                    # Dilate the subcutaneous fat to cover surface skin.
                    dilate_filter = sitk.BinaryDilateImageFilter()
                    dilate_filter.SetForegroundValue(1)  # subcutaneous fat or vertebrae body
                    dilate_filter.SetKernelRadius(self.config.morphology_radius)
                    seg_img = dilate_filter.Execute(seg_img)
                    if "vertebrae_body" in seg_file.name:
                        dilate_filter.SetForegroundValue(2)  # intervertebral disc
                        dilate_filter.SetKernelRadius(self.config.morphology_radius)
                        seg_img = dilate_filter.Execute(seg_img)

                # Convert to array and exclude from uterus mask
                seg_array = sitk.GetArrayFromImage(seg_img)
                uterus_array[seg_array > 0] = 0

                logger.debug(f"Excluded {seg_file.name}")
                
            except Exception as e:
                logger.warning(f"Could not process segmentation {seg_file}: {e}")
        
        # Get reference array for additional filtering
        ref_array = sitk.GetArrayFromImage(reference_img)
        
        # Apply comprehensive filtering
        uterus_array = self._apply_comprehensive_filtering(uterus_array, ref_array)
        
        # Convert back to SimpleITK image
        uterus_mask = sitk.GetImageFromArray(uterus_array)
        uterus_mask.CopyInformation(reference_img)
        
        # Apply advanced refinement techniques
        uterus_mask = self._clean_mask(uterus_mask)
        uterus_mask = self._keep_largest_components(uterus_mask, max_components=self.config.max_components)
        
        logger.info(f"Created refined uterus mask. Remaining voxels: {np.sum(sitk.GetArrayFromImage(uterus_mask) > 0)}")
        
        return uterus_mask
    
    def _clean_mask(self, mask: sitk.Image) -> sitk.Image:
        closing = sitk.VotingBinaryIterativeHoleFillingImageFilter()
        closing.SetRadius(self.config.morphology_radius)
        mask = closing.Execute(mask)
        
        return mask
    
    def _apply_comprehensive_filtering(self, uterus_array: np.ndarray, ref_array: np.ndarray) -> np.ndarray:
        """Apply comprehensive filtering to remove non-uterine tissue"""
        
        # 3. Focus on soft tissue range typical for uterus
        uterus_array[(ref_array < self.config.min_uterus_hu) | (ref_array > self.config.max_uterus_hu)] = 0
        logger.debug(f"After soft tissue filtering: {np.sum(uterus_array)} voxels")
        
        # 4. Apply anatomical constraints based on image dimensions
        z_size, y_size, x_size = uterus_array.shape
        
        # 5. Restrict to central region (exclude peripheral areas)
        uterus_array[:, :, :int(x_size * self.config.x_margin[0])] = 0
        uterus_array[:, :, int(y_size * self.config.x_margin[1]) :] = 0
        uterus_array[:, :int(y_size * self.config.y_margin[0]), :] = 0 
        uterus_array[:, int(y_size * self.config.y_margin[1]) :, :] = 0
        uterus_array[:int(z_size * self.config.z_margin[0]), :, :] = 0
        uterus_array[int(z_size * self.config.z_margin[1]) :, :, :] = 0
        
        logger.debug(f"After anatomical constraints: {np.sum(uterus_array)} voxels")
        
        return uterus_array
    
    def _keep_largest_components(self, mask: sitk.Image, max_components: int = None) -> sitk.Image:
        """Keep only the largest connected components"""
           
        # Connected component labeling
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled_img = cc_filter.Execute(mask)
        
        # Label statistics to get component sizes
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(labeled_img)
        
        # Get labels sorted by size
        labels = label_stats.GetLabels()
        if not labels:
            return mask
        
        # Sort by number of pixels (descending)
        label_sizes = [(label, label_stats.GetNumberOfPixels(label)) for label in labels]
        label_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the largest components
        keep_labels = [label for label, size in label_sizes[:max_components]]
        
        # Create new mask with only selected components
        new_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        new_mask.CopyInformation(mask)
        
        for label in keep_labels:
            # Extract this component
            threshold = sitk.BinaryThresholdImageFilter()
            threshold.SetLowerThreshold(label)
            threshold.SetUpperThreshold(label)
            threshold.SetInsideValue(1)
            threshold.SetOutsideValue(0)
            component = threshold.Execute(labeled_img)
            
            # Add to new mask
            add_filter = sitk.AddImageFilter()
            new_mask = add_filter.Execute(new_mask, component)
        
        logger.info(f"Kept {len(keep_labels)} largest components")
        return new_mask
    
    def _detect_hemorrhage_all_phases(self, nifti_files: Dict[str, Path], 
                                    uterus_masks: Dict[str, sitk.Image], 
                                    output_dir: Path) -> Dict[str, Dict]:
        """Detect hemorrhage in all phases using respective uterus masks"""
        results = {}
        
        hu_thresholds = {
            "non_enhanced": self.config.non_enhanced_hu,
            "arterial": self.config.arterial_hu,
            "portal": self.config.portal_hu
        }
        
        for phase, nifti_path in nifti_files.items():
            logger.info(f"Processing {phase} phase")
            
            # Check if hemorrhage mask already exists
            mask_path = output_dir / f"{phase}_hemorrhage.nii.gz"
            if mask_path.exists():
                logger.info(f"Hemorrhage mask already exists for {phase}, loading existing mask...")
                hemorrhage_mask = sitk.ReadImage(str(mask_path))
                volume_ml = self._calculate_volume(hemorrhage_mask)
                
                results[phase] = {
                    "volume_ml": volume_ml,
                    "hu_threshold": hu_thresholds[phase],
                    "mask_file": mask_path.name
                }
                
                logger.info(f"{phase}: {volume_ml:.2f} mL (existing)")
                continue
            
            # Load image
            img = sitk.ReadImage(str(nifti_path))

            # Gaussian smoothing with configurable sigma
            gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
            gaussian.SetSigma(self.config.region_growing_sigma)
            smoothed = gaussian.Execute(sitk.Cast(img, sitk.sitkFloat32))

            # Apply HU threshold
            min_hu, max_hu = hu_thresholds[phase]
            threshold = sitk.BinaryThresholdImageFilter()
            threshold.SetLowerThreshold(min_hu)
            threshold.SetUpperThreshold(max_hu)
            threshold.SetInsideValue(1)
            threshold.SetOutsideValue(0)
            hemorrhage_mask = threshold.Execute(img)
            
            # Apply phase-specific uterus mask
            multiply = sitk.MultiplyImageFilter()
            hemorrhage_mask = multiply.Execute(hemorrhage_mask, uterus_masks[phase])
            
            # Calculate volume
            volume_ml = self._calculate_volume(hemorrhage_mask)
            
            # Save mask
            sitk.WriteImage(hemorrhage_mask, str(mask_path))
            
            results[phase] = {
                "volume_ml": volume_ml,
                "hu_threshold": hu_thresholds[phase],
                "mask_file": mask_path.name
            }
            
            logger.info(f"{phase}: {volume_ml:.2f} mL")
        
        return results
    
    def _calculate_volume(self, mask: sitk.Image) -> float:
        """Calculate volume in mL"""
        mask_array = sitk.GetArrayFromImage(mask)
        spacing = mask.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        total_voxels = np.sum(mask_array > 0)
        volume_ml = (total_voxels * voxel_volume_mm3) / 1000.0
        return volume_ml
    
    def _analyze_results(self, hemorrhage_results: Dict) -> Dict[str, Any]:
        """Analyze hemorrhage results across phases"""
        volumes = {phase: data["volume_ml"] for phase, data in hemorrhage_results.items()}
        
        # Calculate enhancement
        arterial_enhancement = volumes["arterial"] - volumes["non_enhanced"]
        portal_change = volumes["portal"] - volumes["arterial"]
        
        # Determine active bleeding
        active_bleeding = arterial_enhancement > 5.0  # >5mL enhancement
        
        # Clinical assessment
        baseline_volume = volumes["non_enhanced"]
        severity = self.config.get_severity(baseline_volume)
        
        # Intervention recommendation
        needs_intervention = (
            baseline_volume > self.config.moderate_threshold or 
            active_bleeding
        )
        
        return {
            "volumes": volumes,
            "arterial_enhancement": arterial_enhancement,
            "portal_change": portal_change,
            "active_bleeding": str(active_bleeding),
            "severity": severity,
            "needs_intervention": str(needs_intervention),
            "peak_phase": max(volumes, key=volumes.get)
        }
    
    def _cleanup(self):
        """Clean up temporary files"""
        try:
            self.dicom_processor.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
