import h5py
import numpy as np

def extract_cell_timetrace(data_path, cell_id):
    with h5py.File(data_path, 'r') as f:
        cell_time_trace = np.array(f['timeTraces']['100']).T
    
    return cell_time_trace[cell_id]
