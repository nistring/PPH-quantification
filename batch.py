import argparse
import json
import logging
import sys
import time
from pathlib import Path
import pandas as pd

from config import Config
from analyzer import SimplePPHAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_patients(input_dir: Path):
    """Find valid patient directories"""
    patients = []
    for patient_dir in input_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        
        # Check for required series
        required_series = ["201", "601", "701"]
        if all((patient_dir / series).exists() for series in required_series):
            patients.append(patient_dir)
            logger.info(f"Found patient: {patient_dir.name}")
    
    return patients

def process_batch(patients, output_dir, config):
    """Process patients sequentially"""
    analyzer = SimplePPHAnalyzer(config)
    results = []
    errors = []
    
    for i, patient_dir in enumerate(patients, 1):
        logger.info(f"Processing {i}/{len(patients)}: {patient_dir.name}")
        
        try:
            patient_output = output_dir / patient_dir.name
            result = analyzer.analyze_patient(patient_dir, patient_output)
            results.append(result)
            
        except Exception as e:
            error_info = {
                "patient": patient_dir.name,
                "error": str(e)
            }
            errors.append(error_info)
            logger.error(f"Failed to process {patient_dir.name}: {e}")
    
    return results, errors

def main():
    parser = argparse.ArgumentParser(description="Batch PPH Processing")
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Directory containing patient subdirectories")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Output directory")
    parser.add_argument("--fast", action="store_true",
                       help="Use fast mode")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find patients
    patients = find_patients(input_dir)
    if not patients:
        logger.error("No valid patients found")
        sys.exit(1)
    
    logger.info(f"Found {len(patients)} patients")
    
    # Initialize configuration
    config = Config()
    config.use_fast_mode = args.fast
    
    # Process batch
    start_time = time.time()
    results, errors = process_batch(patients, output_dir, config)
    total_time = time.time() - start_time
    
    # Create summary
    if results:
        df = pd.DataFrame([
            {
                "patient": r["patient_name"],
                "non_enhanced_ml": r["hemorrhage_volumes"]["non_enhanced"],
                "arterial_ml": r["hemorrhage_volumes"]["arterial"],
                "portal_ml": r["hemorrhage_volumes"]["portal"],
                "severity": r["analysis"]["severity"],
                "active_bleeding": r["analysis"]["active_bleeding"],
                "needs_intervention": r["analysis"]["needs_intervention"]
            }
            for r in results
        ])
        
        # Save summary
        df.to_csv(output_dir / "batch_summary.csv", index=False)
        
        # Print summary
        print(f"\nBatch processing completed in {total_time:.1f}s")
        print(f"Successful: {len(results)}, Failed: {len(errors)}")
        print(f"\nSeverity distribution:")
        print(df["severity"].value_counts())
        print(f"Active bleeding cases: {df['active_bleeding'].sum()}")
        print(f"Intervention needed: {df['needs_intervention'].sum()}")
    
    # Save errors
    if errors:
        with open(output_dir / "errors.json", 'w') as f:
            json.dump(errors, f, indent=2)

if __name__ == "__main__":
    main()
