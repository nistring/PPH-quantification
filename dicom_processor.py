import logging
from pathlib import Path
from typing import Optional, Union
import warnings

import pydicom
import SimpleITK as sitk
import dicom2nifti

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DICOMProcessor:
    """
    Handles DICOM file processing, conversion, and preprocessing for hemorrhage analysis
    """

    def __init__(self, temp_dir: str = "temp"):
        """
        Initialize DICOM processor

        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def read_dicom_file(self, dicom_path: Union[str, Path]) -> pydicom.Dataset:
        """
        Read a single DICOM file

        Args:
            dicom_path: Path to DICOM file

        Returns:
            DICOM dataset
        """
        try:
            ds = pydicom.dcmread(str(dicom_path))
            logger.info(f"Successfully read DICOM file: {dicom_path}")
            return ds
        except Exception as e:
            logger.error(f"Error reading DICOM file {dicom_path}: {e}")
            raise

    def dicom_to_nifti(self, dicom_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Convert DICOM file to NIfTI format

        Args:
            dicom_path: Path to DICOM file or directory containing DICOM series
            output_path: Output path for NIfTI file

        Returns:
            Path to converted NIfTI file
        """
        dicom_path = Path(dicom_path)

        if output_path is None:
            if dicom_path.is_dir():
                # Use directory name for series
                output_path = self.temp_dir / f"{dicom_path.name}.nii.gz"
            else:
                # Use file stem for single file
                output_path = self.temp_dir / f"{dicom_path.stem}.nii.gz"
        else:
            output_path = Path(output_path)

        try:
            if dicom_path.is_file():
                # Single DICOM file - read with SimpleITK
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(dicom_path))
                image = reader.Execute()

                # Write as NIfTI
                writer = sitk.ImageFileWriter()
                writer.SetFileName(str(output_path))
                writer.Execute(image)

            elif dicom_path.is_dir():
                # DICOM series directory - use dicom2nifti
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dicom2nifti.dicom_series_to_nifti(
                        str(dicom_path), 
                        str(output_path), 
                        reorient_nifti=True
                    )
            else:
                raise ValueError(f"Invalid DICOM path: {dicom_path}")

            logger.info(f"Successfully converted DICOM to NIfTI: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error converting DICOM to NIfTI: {e}")
            raise

    def cleanup_temp_files(self):
        """Remove temporary files"""
        for temp_file in self.temp_dir.glob("*"):
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")
