import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd

from config import Config
from analyzer import PPHAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_patient(patient_data):
    """Process a single patient - designed to work with multiprocessing"""
    patient_dir, output_dir, config_dict = patient_data
    
    try:
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        analyzer = PPHAnalyzer(config)
        patient_output = output_dir / patient_dir.name
        result = analyzer.analyze_patient(patient_dir, patient_output)
        return result, None
        
    except Exception as e:
        logger.error(f"Failed to process {patient_dir.name}: {e}")
        return None, {"patient": patient_dir.name, "error": str(e)}

class PPHBatchProcessor:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.analyzer = PPHAnalyzer(self.config)
    
    def find_patients(self, input_dir: Path):
        """Find patient directories with Arterial/Portal subdirectories"""
        patients = []
        for patient_dir in input_dir.iterdir():
            if patient_dir.is_dir() and all((patient_dir / series).exists() 
                                          for series in ["Arterial", "Portal"]):
                patients.append(patient_dir)
                logger.info(f"Found patient: {patient_dir.name}")
        return patients
    
    def process_batch(self, patients, output_dir, parallel=True, max_workers=None):
        """Process patients with parallel or sequential processing"""
        if parallel and len(patients) > 1:
            return self._process_parallel(patients, output_dir, max_workers)
        else:
            return self._process_sequential(patients, output_dir)
    
    def _process_parallel(self, patients, output_dir, max_workers):
        """Process patients in parallel"""
        if max_workers is None:
            max_workers = min(len(patients), multiprocessing.cpu_count())
        
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        config_dict = {attr: getattr(self.config, attr) 
                      for attr in dir(self.config) 
                      if not attr.startswith('_') and not callable(getattr(self.config, attr))}
        
        patient_data = [(patient_dir, output_dir, config_dict) for patient_dir in patients]
        results, errors = [], []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_patient = {executor.submit(process_single_patient, data): data[0] 
                               for data in patient_data}
            
            for i, future in enumerate(as_completed(future_to_patient), 1):
                patient_dir = future_to_patient[future]
                logger.info(f"Completed {i}/{len(patients)}: {patient_dir.name}")
                
                try:
                    result, error = future.result()
                    if result: results.append(result)
                    if error: errors.append(error)
                except Exception as e:
                    errors.append({"patient": patient_dir.name, "error": str(e)})
                    logger.error(f"Failed to process {patient_dir.name}: {e}")
        
        return results, errors
    
    def _process_sequential(self, patients, output_dir):
        """Sequential processing fallback"""
        results, errors = [], []
        
        for i, patient_dir in enumerate(patients, 1):
            logger.info(f"Processing {i}/{len(patients)}: {patient_dir.name}")
            try:
                patient_output = output_dir / patient_dir.name
                result = self.analyzer.analyze_patient(patient_dir, patient_output)
                results.append(result)
            except Exception as e:
                errors.append({"patient": patient_dir.name, "error": str(e)})
                logger.error(f"Failed to process {patient_dir.name}: {e}")
        
        return results, errors

def create_summary_report(results, output_dir):
    """Create CSV summary report"""
    if not results:
        return
    
    df = pd.DataFrame([{
        "patient": r["patient_name"],
        "arterial_ml": r["hemorrhage_volumes"]["arterial"],
        "portal_ml": r["hemorrhage_volumes"]["portal"],
        "change_ml": r["analysis"]["portal_arterial_change"],
        "severity": r["analysis"]["severity"],
        "active_bleeding": r["analysis"]["active_bleeding"],
        "intervention_needed": r["analysis"]["needs_intervention"],
        "processing_time": f"{r['processing_time']:.1f}s"
    } for r in results])
    
    summary_path = output_dir / "batch_summary.csv"
    df.to_csv(summary_path, index=False)
    
    print(f"\nBATCH SUMMARY - {len(results)} patients processed")
    print(f"Active bleeding: {df['active_bleeding'].str.contains('True').sum()}")
    print(f"Intervention needed: {df['intervention_needed'].str.contains('True').sum()}")
    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="PPH Batch Processing")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Use fast mode")
    parser.add_argument("--sequential", action="store_true", help="Sequential processing")
    parser.add_argument("--workers", "-w", default=12, type=int, help="Number of workers")
    
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config()
    config.use_fast_mode = args.fast
    processor = PPHBatchProcessor(config)
    
    patients = processor.find_patients(input_dir)
    if not patients:
        logger.error("No valid patients found")
        sys.exit(1)
    
    logger.info(f"Found {len(patients)} patients")
    
    start_time = time.time()
    results, errors = processor.process_batch(patients, output_dir, 
                                            not args.sequential, args.workers)
    
    create_summary_report(results, output_dir)
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    if errors:
        with open(output_dir / "errors.json", 'w') as f:
            json.dump(errors, f, indent=2)

if __name__ == "__main__":
    main()
