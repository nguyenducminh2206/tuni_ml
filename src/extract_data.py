from processing_file import return_simulation_info, read_file
import pandas as pd
import h5py
import numpy as np
import os
import re


def build_df(data_path):
    simulation_id, file = return_simulation_info(data_path)
    rows = []

    with h5py.File(file, 'r') as f:
        # Extract the ids of 1000 samples in the simulation
        raw_ids = f['sim_ids'][()]
        sample_ids = [x.decode('utf-8') for x in raw_ids]

        # Extract the features name
        feature_names = list(f['features'].keys())

        for i, full_sample_ids in enumerate(sample_ids):
            sample_keys = str(i + 1)
            row_data = {'sample_id': full_sample_ids}

            time_trace = np.array(f['timeTraces'][sample_keys])
            row_data['time_trace'] = time_trace

            for feature in feature_names:
                sample_info = np.array(f['features'][feature][sample_keys])
                row_data[feature] = sample_info

            rows.append(row_data)
    
    df = pd.DataFrame(rows)

    return simulation_id, df


def load_simulation_data(data_folder: str) -> pd.DataFrame:
    """
    For every file in the folder extract sim_ids, time_trace, noise and distance_to_target
    and magnitude.
    
    """
    rows = []

    file_paths = read_file(data_folder)
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            raw_ids = f['sim_ids'][()]
            sample_ids = [x.decode('utf-8') for x in raw_ids]

            magnitude = f['stim']['stimulusMagnitude'][()]
            dist_to_target = np.array(f['tissue']['distanceToTarget'][()])

            for i, sid in enumerate(sample_ids):
                key = str(i+1)
                time_trace = np.array(f['timeTraces'][key])

                rows.append({
                    'sample_id': sid,
                    'time_trace': time_trace,
                    'magnitude': magnitude,
                    'distance_to_target': dist_to_target
                })
    df = pd.DataFrame(rows, columns=['sample_id', 'time_trace', 'magnitude', 'distance_to_target'])
    df.to_pickle('src/simulation_data.pkl')
    return df


def build_cellwise_df(data_path):
    """
    For each .h5 file, extracts cellwise time traces and
    their corresponding distance to target.

    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            simulation_id = f['sim_ids'][()][0].decode('utf-8') # First simulation_id
            # Extract the first sample only
            time_traces = np.array(f['timeTraces']['1']).T
            distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
            # Number of cells
            n_cells = time_traces.shape[0]

            for cell_id in range(n_cells):
                rows.append({
                    'simulation_id': simulation_id,
                    'cell_id': cell_id,
                    'time_trace': time_traces[cell_id],
                    'dis_to_target': distance_to_target[cell_id],
                    'simulation_file': os.path.basename(file_path)
                })
    df = pd.DataFrame(rows)
    return df

            
def build_cellwise_all_df(data_path):
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            simulation_ids = [x.decode('utf-8') for x in f['sim_ids'][()]]  # list of all simulation IDs

            # Iterate over all sample keys in timeTraces
            for sample_idx, sample_key in enumerate(f['timeTraces'].keys()):  # e.g., "1", "2", ..., "1000"
                time_traces = np.array(f['timeTraces'][sample_key]).T  # (nCells, nTimePoints)
                distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
                n_cells = time_traces.shape[0]

                for cell_id in range(n_cells):
                    rows.append({
                        'simulation_id': simulation_ids[sample_idx],  # ID for this sample
                        'sample_key': sample_key,
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'simulation_file': os.path.basename(file_path)
                    })
                    
    df = pd.DataFrame(rows)
    return df


def build_cellwise_df_10samplesperfile(data_path):
    """
    For each .h5 file, extracts cellwise time traces and their corresponding distance to target,
    for the first 10 samples in the file.
    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            simulation_ids = [x.decode('utf-8') for x in f['sim_ids'][()]]
            sample_keys = sorted(f['timeTraces'].keys(), key=lambda x: int(x))[:10]  # first 10 samples

            for sample_idx, sample_key in enumerate(sample_keys):
                time_traces = np.array(f['timeTraces'][sample_key]).T
                distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
                n_cells = time_traces.shape[0]

                for cell_id in range(n_cells):
                    rows.append({
                        'simulation_id': simulation_ids[sample_idx],
                        'sample_key': sample_key,
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'simulation_file': os.path.basename(file_path)
                    })
    df = pd.DataFrame(rows)
    return df


def build_cellwise_df_100samplesperfile(data_path):
    """
    For each .h5 file, extracts cellwise time traces and their corresponding distance to target,
    for the first 100 samples in the file.
    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            simulation_ids = [x.decode('utf-8') for x in f['sim_ids'][()]]
            sample_keys = sorted(f['timeTraces'].keys(), key=lambda x: int(x))[:100]  # first 100 samples

            for sample_idx, sample_key in enumerate(sample_keys):
                time_traces = np.array(f['timeTraces'][sample_key]).T
                distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
                n_cells = time_traces.shape[0]

                for cell_id in range(n_cells):
                    rows.append({
                        'simulation_id': simulation_ids[sample_idx],
                        'sample_key': sample_key,
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'simulation_file': os.path.basename(file_path)
                    })
    df = pd.DataFrame(rows)
    return df


def build_features_10samples_df(data_path):
    """
    For each .h5 file, for the first 10 samples, extract time traces, distance to target,
    and all feature values (per feature, per sample, per cell), filling missing with NaN.
    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            simulation_ids = [x.decode('utf-8') for x in f['sim_ids'][()]]
            sample_keys = sorted(f['timeTraces'].keys(), key=lambda x: int(x))[:10]  # first 10 samples
            feature_names = list(f['features'].keys())

            for sample_idx, sample_key in enumerate(sample_keys):
                time_traces = np.array(f['timeTraces'][sample_key]).T  # (n_cells, n_timepoints)
                distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
                n_cells = time_traces.shape[0]

                # For each feature, get vector for this sample_key (handle missing and shape)
                feature_vectors = {}
                for feature in feature_names:
                    if sample_key in f['features'][feature]:
                        vector = np.array(f['features'][feature][sample_key])  # shape (1, 25) or (0, 25)
                        if vector.shape == (1, n_cells):
                            feature_vectors[feature] = vector[0, :]  # shape (25,)
                        else:
                            # Missing, empty, or wrong shape: fill with NaN
                            feature_vectors[feature] = np.full(n_cells, np.nan)
                    else:
                        feature_vectors[feature] = np.full(n_cells, np.nan)

                for cell_id in range(n_cells):
                    row = {
                        'simulation_id': simulation_ids[sample_idx],
                        'sample_key': sample_key,
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'simulation_file': os.path.basename(file_path)
                    }
                    for feature in feature_names:
                        row[feature] = feature_vectors[feature][cell_id]
                    rows.append(row)
    df = pd.DataFrame(rows)
    return df


def extract_noise(filename):
    match = re.search(r'noise[_\-]?([0-9.]+)', filename)
    return float(match.group(1) if match else None)


def main():
    data_path = 'data'
    build_features_10samples_df(data_path)


if __name__ == "__main__":
    main()
