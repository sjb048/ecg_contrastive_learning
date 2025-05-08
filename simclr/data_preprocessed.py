import pandas as pd
import numpy as np
import os
import wfdb
import torch

def extract_signals_to_numpy(csv_file, base_dir, filename_col='filename_hr', max_files=5000, segment_length=1000, out_path='ecg_signals_simclr_12lead.npy'):
    print(f"Extracting signals using {filename_col} (sampling rate: {'500 Hz' if 'hr' in filename_col else '100 Hz'})")
    df = pd.read_csv(csv_file).iloc[:max_files]
    signals = []
    ecg_ids = []
    for _, row in df.iterrows():
        full_path = os.path.join(base_dir, row[filename_col])
        try:
            record = wfdb.rdrecord(full_path)
            sig = record.p_signal[:segment_length, :]  # Extract all 12 leads
            if sig.shape[0] < segment_length:
                print(f"Signal too short in {full_path}, padding to {segment_length}.")
                sig = np.pad(sig, ((0, segment_length - sig.shape[0]), (0, 0)), mode='constant', constant_values=0)
            signals.append(sig)
            ecg_ids.append(row['ecg_id'])
        except Exception as e:
            print(f"Skipping {full_path}: {e}")
            continue
    if not signals:
        raise RuntimeError("No valid signals found.")
    arr = np.stack(signals)  # shape: (num_samples, segment_length, 12)
    np.save(out_path, arr)
    np.save(out_path.replace('.npy', '_ids.npy'), np.array(ecg_ids))
    loaded_arr = np.load(out_path)
    if loaded_arr.shape != arr.shape:
        raise RuntimeError(f"Shape mismatch in saved file {out_path}: expected {arr.shape}, got {loaded_arr.shape}")
    print(f"Saved and verified {arr.shape} to {out_path} and IDs to {out_path.replace('.npy', '_ids.npy')}")

# Regenerate the data file with all 12 leads
extract_signals_to_numpy(
    csv_file='../data/ptbxl_database.csv',
    base_dir='../data',
    filename_col='filename_hr',
    max_files=5000,
    segment_length=1000,
    out_path='ecg_signals_simclr_12lead.npy'
)
