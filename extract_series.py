import os
import pydicom
import shutil

# Set your data directory and output directory
DATA_DIR = 'data'
OUTPUT_DIR = 'extracted_series'
SERIES_NUMBERS = {'201', '601', '701'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Traverse each patient directory
for patient in os.listdir(DATA_DIR):
    patient_path = os.path.join(DATA_DIR, patient)
    if not os.path.isdir(patient_path):
        continue
    # Prepare output subfolders for each series
    for series_number in SERIES_NUMBERS:
        os.makedirs(os.path.join(OUTPUT_DIR, patient, series_number), exist_ok=True)
    # Traverse DICOM files
    for root, _, files in os.walk(patient_path):
        for file in files:
            if not file.lower().endswith('.dcm'):
                continue
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                series_number = str(getattr(ds, 'SeriesNumber', ''))
                if series_number in SERIES_NUMBERS:
                    dest_dir = os.path.join(OUTPUT_DIR, patient, series_number)
                    shutil.copy(file_path, dest_dir)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
print('Extraction complete.')
