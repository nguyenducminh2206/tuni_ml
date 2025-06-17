from processing_file import return_simulation_info, read_file
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


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

def plot_timetrace(data_path):
    simulation_id, df = build_df(data_path)
    
    # Plot simulation
    sim_id = df['sample_id'].iloc[0]
    time_trace = df['time_trace'].iloc[0]

    plt.figure(figsize=(10,6))

    n_sample = time_trace[:,6:13]
    for cell_id in range(n_sample.shape[1]):
        plt.plot(time_trace[:, cell_id], label=f'cell {cell_id + 1}')
    
    plt.legend(
        title='Cell',
        ncol=2,
        loc='upper right'
    )

    plt.title(f'time trace for sample {sim_id}')
    plt.xlabel('time index')
    plt.ylabel('signal')
    plt.tight_layout()
    plt.grid()
    plt.show()

    return simulation_id

def load_noise_folder_data(noise_folder: str) -> pd.DataFrame:
    """
    For every file in the folder noise_n, extract sim_ids, time_trace and magnitude.
    
    """
    rows = []

    noise = float(os.path.basename(noise_folder).split('_')[1])

    for file_name in read_file(noise_folder):
        full_path = os.path.join(noise_folder, file_name)

        with h5py.File(full_path, 'r') as f:
            # extract sample ids
            raw_ids = f['sim_ids'][()]
            sample_ids = [x.decode('utf-8') for x in raw_ids]

            # extract magnitude
            mags = f['stim']['stimulusMagnitude'][()]

            # iterate through each sample
            for i, sid in enumerate(sample_ids):
                key = str(i + 1)
                tt = np.array(f['timeTraces'][key])
                
                rows.append({
                    'sample_id': sid,
                    'time_trace': tt,
                    'noise': noise,
                    'magnitude': mags
                })
    
    df = pd.DataFrame(rows, columns=['sample_id', 'time_trace','noise', 'magnitude'])

    return df

def concatenate_df():
    df_1 = load_noise_folder_data('data/noise_0.01')
    df_2 = load_noise_folder_data('data/noise_0.02')

    df = pd.concat([df_1, df_2], axis = 0, ignore_index=True)

    return df