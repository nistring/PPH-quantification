import os
import pydicom
import shutil
from collections import defaultdict
from multiprocessing import Pool
import multiprocessing as mp

DATA_DIR = 'data'
OUTPUT_DIR = 'extracted_series'
PHASE_KEYWORDS = {
    'arterial': ['arter', 'ap'],
    'portal': ['portal', 'pvp']
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def identify_phase(description):
    """Identify phase from series description"""
    if not description:
        return None
    desc_lower = description.lower()
    for phase, keywords in PHASE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return phase
    return None

def get_patient_series(patient_path):
    """Get best series for each phase"""
    series_data = defaultdict(list)
    
    for root, _, files in os.walk(patient_path):
        for file in files:
            if not file.lower().endswith('.dcm'):
                continue
            
            try:
                ds = pydicom.dcmread(os.path.join(root, file), stop_before_pixels=True)
                series_num = str(getattr(ds, 'SeriesNumber', ''))
                description = f"{getattr(ds, 'SeriesDescription', '')} {getattr(ds, 'ProtocolName', '')}".strip()
                phase = identify_phase(description)
                
                if series_num and phase:
                    series_data[phase].append({
                        'series': series_num,
                        'file': os.path.join(root, file),
                        'desc': getattr(ds, 'SeriesDescription', '')
                    })
            except:
                continue
    
    # Select series with most files for each phase
    result = {}
    for phase, files in series_data.items():
        series_groups = defaultdict(list)
        for f in files:
            series_groups[f['series']].append(f)
        
        if series_groups:
            best_series = max(series_groups.values(), key=len)
            result[phase] = best_series
    
    return result

def process_patient(args):
    """Process a single patient - designed for multiprocessing"""
    patient, total_patients, patient_index = args
    patient_path = os.path.join(DATA_DIR, patient)
    
    series = get_patient_series(patient_path)
    if not series:
        return f"{patient_index}/{total_patients}: {patient} - No phases found"
    
    patient_output = os.path.join(OUTPUT_DIR, patient.split()[0].zfill(8))
    os.makedirs(patient_output, exist_ok=True)
    
    results = [f"{patient_index}/{total_patients}: {patient}"]
    
    for phase, files in series.items():
        phase_dir = os.path.join(patient_output, phase.capitalize())
        os.makedirs(phase_dir, exist_ok=True)
        
        for file_info in files:
            dest = os.path.join(phase_dir, os.path.basename(file_info['file'])[-8:])
            if not os.path.exists(dest):
                shutil.copy(file_info['file'], dest)
        
        results.append(f"  {phase}: {len(files)} files")
    
    return '\n'.join(results)

if __name__ == '__main__':
    # Get number of processes
    nprocs = mp.cpu_count()  # or set to desired number
    
    # Process patients
    patients = [p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))]
    print(f"Processing {len(patients)} patients with {nprocs} processes...")
    
    # Prepare arguments for multiprocessing
    patient_args = [(patient, len(patients), i+1) for i, patient in enumerate(patients)]
    
    # Process in parallel
    with Pool(processes=nprocs) as pool:
        results = pool.map(process_patient, patient_args)
    
    # Print results
    for result in results:
        print(result)
    
    print("Extraction complete!")
