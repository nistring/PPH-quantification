# Simplified Postpartum Hemorrhage Quantification Tool

## Overview

This Python project provides a simplified yet effective approach to automated quantification of postpartum hemorrhage from **three-phase CT series** (201=non-enhanced, 601=arterial, 701=portal phase). The tool uses TotalSegmentator for anatomical exclusion to create uterine masks and applies Hounsfield Unit (HU) thresholding for hemorrhage detection.

## Methodology

The tool implements a straightforward three-phase analysis approach:

1. **DICOM to NIfTI Conversion**: Converts all three phases to NIfTI format
2. **Anatomical Masking**: Uses TotalSegmentator to exclude non-uterine structures (organs, subcutaneous fat, vertebrae)
3. **Phase-Specific Analysis**: Applies different HU thresholds for each phase
4. **Volume Calculation**: Measures hemorrhage volumes within the uterine region
5. **Clinical Assessment**: Provides severity classification and intervention recommendations

### Hounsfield Unit Thresholds

The tool uses simplified HU ranges for hemorrhage detection:

**Non-Enhanced Phase (Series 201):**
- **Hemorrhage detection**: 50-75 HU

**Arterial Phase (Series 601):**
- **Active bleeding**: 85-370 HU

**Portal Phase (Series 701):**
- **Enhancement/pooling**: 85-370 HU

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

### Step 3: Verify Installation

```bash
# Test TotalSegmentator installation
TotalSegmentator --help

# Test the main script
python main.py --help
```

## Usage

### Basic Three-Phase Analysis

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
- `--patient-dir, -p`: Path to patient directory containing 201/, 601/, 701/ subdirectories
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
│   ├── 201/  (non-enhanced CT)
│   │   ├── Patient001_0001.dcm
│   │   ├── Patient001_0002.dcm
│   │   └── ...
│   ├── 601/  (arterial phase)
│   │   ├── Patient001_0001.dcm
│   │   ├── Patient001_0002.dcm
│   │   └── ...
│   └── 701/  (portal phase)
│       ├── Patient001_0001.dcm
│       ├── Patient001_0002.dcm
│       └── ...
├── Patient002/
│   ├── 201/
│   ├── 601/
│   └── 701/
└── Patient003/
    ├── 201/
    ├── 601/
    └── 701/
```

## Output

The tool generates simplified outputs in the specified directory:

### Files Structure

```
results/Patient001/
├── results.json                         # Complete analysis results
├── non_enhanced.nii.gz                  # Converted non-enhanced CT
├── arterial.nii.gz                      # Converted arterial phase CT  
├── portal.nii.gz                        # Converted portal phase CT
├── uterus_mask_non_enhanced.nii.gz      # Uterus mask for non-enhanced
├── uterus_mask_arterial.nii.gz          # Uterus mask for arterial
├── uterus_mask_portal.nii.gz            # Uterus mask for portal
├── non_enhanced_hemorrhage.nii.gz       # Hemorrhage mask for non-enhanced
├── arterial_hemorrhage.nii.gz           # Hemorrhage mask for arterial
├── portal_hemorrhage.nii.gz             # Hemorrhage mask for portal
├── total_non_enhanced.nii               # TotalSegmentator output
├── total_arterial.nii                   # TotalSegmentator output
├── total_portal.nii                     # TotalSegmentator output
├── tissue_types_non_enhanced.nii        # TotalSegmentator tissue types
├── tissue_types_arterial.nii            # TotalSegmentator tissue types
├── tissue_types_portal.nii              # TotalSegmentator tissue types
├── vertebrae_body_non_enhanced.nii      # TotalSegmentator vertebrae
├── vertebrae_body_arterial.nii          # TotalSegmentator vertebrae
├── vertebrae_body_portal.nii            # TotalSegmentator vertebrae
└── batch_summary.csv                    # (For batch processing)
```

### Analysis Results JSON

The simplified JSON output includes:

```json
{
  "patient_name": "Patient001",
  "processing_time": 180.5,
  "hemorrhage_volumes": {
    "non_enhanced": 125.6,
    "arterial": 185.2,
    "portal": 165.8
  },
  "uterus_mask_volumes": {
    "non_enhanced": 450.2,
    "arterial": 448.1,
    "portal": 449.3
  },
  "analysis": {
    "volumes": {
      "non_enhanced": 125.6,
      "arterial": 185.2,
      "portal": 165.8
    },
    "arterial_enhancement": 59.6,
    "portal_change": -19.4,
    "active_bleeding": "True",
    "severity": "Moderate",
    "needs_intervention": "True",
    "peak_phase": "arterial"
  },
  "output_dir": "results/Patient001"
}
```

**Disclaimer**: This tool is for research purposes only and is not intended for clinical diagnosis or treatment decisions.
