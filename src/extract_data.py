from processing_file import return_simulation_info
import pandas as pd
import h5py
import numpy as np

data_path = 'data'

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

            for feature in feature_names:
                sample_info = np.array(f['features'][feature][sample_keys])
                row_data[feature] = sample_info

            rows.append(row_data)
    
    df = pd.DataFrame(rows)

    return simulation_id, df
