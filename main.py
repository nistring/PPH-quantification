import argparse
import logging
import sys
from pathlib import Path

from config import Config
from analyzer import PPHAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Simplified PPH Quantification")
    parser.add_argument("--patient-dir", "-p", required=True,
                       help="Patient directory")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory")
    parser.add_argument("--fast", action="store_true",
                       help="Use fast mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    patient_dir = Path(args.patient_dir)
    output_dir = Path(args.output)
    
    if not patient_dir.exists():
        logger.error(f"Patient directory not found: {patient_dir}")
        sys.exit(1)
    
    # Initialize configuration
    config = Config()
    config.use_fast_mode = args.fast
    
    # Initialize analyzer
    analyzer = PPHAnalyzer(config)
    
    results = analyzer.analyze_patient(patient_dir, output_dir)

if __name__ == "__main__":
    main()
