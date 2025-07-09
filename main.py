import argparse
import logging
import sys
from pathlib import Path

from config import Config
from analyzer import SimplePPHAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Simplified PPH Quantification")
    parser.add_argument("--patient-dir", "-p", required=True,
                       help="Patient directory with 201/, 601/, 701/ subdirectories")
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
    analyzer = SimplePPHAnalyzer(config)
    
    try:
        # Run analysis
        results = analyzer.analyze_patient(patient_dir, output_dir)
        
        # Print results
        print("\n" + "="*60)
        print("PPH ANALYSIS RESULTS")
        print("="*60)
        print(f"Patient: {results['patient_name']}")
        print(f"Processing time: {results['processing_time']:.1f}s")
        print("\nHemorrhage volumes:")
        for phase, volume in results['hemorrhage_volumes'].items():
            print(f"  {phase:12}: {volume:6.2f} mL")
        
        analysis = results['analysis']
        print(f"\nAnalysis:")
        print(f"  Arterial enhancement: {analysis['arterial_enhancement']:+.2f} mL")
        print(f"  Portal change:        {analysis['portal_change']:+.2f} mL")
        print(f"  Active bleeding:      {analysis['active_bleeding']}")
        print(f"  Severity:            {analysis['severity']}")
        print(f"  Needs intervention:  {analysis['needs_intervention']}")
        print(f"  Peak phase:          {analysis['peak_phase']}")
        
        print(f"\nResults saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
