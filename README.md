# Simplified Postpartum Hemorrhage Quantification Tool

## Overview

This Python project provides a simplified yet effective approach to automated quantification of postpartum hemorrhage from **two-phase CT series** (arterial, portal phase). The tool uses TotalSegmentator for anatomical exclusion to create a common uterine mask and applies Hounsfield Unit (HU) thresholding for hemorrhage detection.

## Methodology

The tool implements a straightforward two-phase analysis approach:

1. **DICOM to NIfTI Conversion**: Converts arterial and portal phases to NIfTI format
2. **Common Anatomical Masking**: Uses TotalSegmentator to exclude non-uterine structures for both phases, then creates intersection mask
3. **Morphological Processing**: Applies hole filling and smoothing operations to the masks
4. **Two-Phase Analysis**: Applies different HU thresholds for each phase (no lower limit, upper limits of 150 HU for arterial, 100 HU for portal)
5. **Volume Calculation**: Measures hemorrhage volumes within the common uterine region
6. **Clinical Assessment**: Provides severity classification and intervention recommendations

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster processing)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd pph-quantification

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```
To use [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), a license for non-commercial use is required. Please refer to the link for more information.

### Step 3: Verify Installation

```bash
# Test TotalSegmentator installation
TotalSegmentator --help

# Test the main script
python main.py --help
```

## Usage

### Basic Two-Phase Analysis

```bash
# Analyze patient directory with extracted series
python main.py --patient-dir extracted_series/Patient001/ --output results/Patient001/
```

### Batch Processing

```bash
# Process all patients in extracted_series directory
python batch.py --input-dir extracted_series/ --output-dir batch_results/
```

### Command Line Options

**Main Script (main.py):**
- `--patient-dir, -p`: Path to patient directory containing 601/, 701/ subdirectories
- `--output, -o`: Output directory for results (default: "output")
- `--fast`: Use fast mode for TotalSegmentator (lower resolution, faster processing)
- `--verbose, -v`: Enable verbose logging

**Batch Processing Script (batch.py):**
- `--input-dir, -i`: Directory containing patient subdirectories with extracted series
- `--output-dir, -o`: Output directory for batch results
- `--fast`: Use fast mode for TotalSegmentator

## Input Requirements

### Expected Directory Structure

After running the series extraction script, your data should be organized as:

```
extracted_series/
├── Patient001/
│   ├── Arterial/
│   │   ├── Patient001_0001.dcm
│   │   ├── Patient001_0002.dcm
│   │   └── ...
│   └── Portal/
│       ├── Patient001_0001.dcm
│       ├── Patient001_0002.dcm
│       └── ...
├── Patient002/
│   ├── Arterial/
│   └── Portal/
└── Patient003/
    ├── Arterial/
    └── Portal/
```

## Output

The tool generates simplified outputs in the specified directory:

### Files Structure

```
results/Patient001/
├── results.json                         # Complete analysis results
├── arterial.nii.gz                      # Converted arterial phase CT  
├── portal.nii.gz                        # Converted portal phase CT
├── common_uterus_mask.nii.gz             # Common uterus mask (intersection)
├── uterus_mask_arterial.nii.gz           # Individual arterial uterus mask
├── uterus_mask_portal.nii.gz             # Individual portal uterus mask
├── arterial_hemorrhage.nii.gz            # Hemorrhage mask for arterial
├── portal_hemorrhage.nii.gz              # Hemorrhage mask for portal
├── total_arterial.nii                    # TotalSegmentator output
├── total_portal.nii                      # TotalSegmentator output
├── tissue_types_arterial.nii             # TotalSegmentator tissue types
├── tissue_types_portal.nii               # TotalSegmentator tissue types
├── vertebrae_body_arterial.nii           # TotalSegmentator vertebrae
├── vertebrae_body_portal.nii             # TotalSegmentator vertebrae
└── batch_summary.csv                     # (For batch processing)
```

### Analysis Results JSON

The simplified JSON output includes:

```json
{
  "patient_name": "Patient001",
  "processing_time": 180.5,
  "hemorrhage_volumes": {
    "arterial": 185.2,
    "portal": 165.8
  },
  "uterus_mask_volume": 449.3,
}
```

**Disclaimer**: This tool is for research purposes only and is not intended for clinical diagnosis or treatment decisions