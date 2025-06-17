import os
import re
import h5py

def read_file(folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    return files

def parse_simulation(filename):
    """
    Extract simulation parameters from the file name
    
    """
    pattern = (
        r"stimMag_(?P<stimMag>\d+(?:\.\d+)?)_"
        r"beta_(?P<beta>\d+(?:\.\d+)?)_"
        r"noise_(?P<noise>\d+(?:\.\d+)?)_"
        r"kcross_(?P<kcross>\d+(?:\.\d+)?)_"
        r"nSamples_(?P<nSamples>\d+(?:\.\d+)?)"
    )

    match = re.search(pattern, filename)

    if match:
        return {k: float(v) for k, v in match.groupdict().items()}
    
    return None

def find_file(data_path, user_params):
    """
    Read all the file names in folder data and return the matching
    with user inputs
    
    """
    files = read_file(data_path)
    for file_name in files:
        if file_name.endswith('.h5'):
            params = parse_simulation(file_name)
            if params and all(abs(params[key] - user_params[key]) < 1e-6 for key in user_params):
                return os.path.join(data_path, file_name)
    return None

def read_user_input():
    
    print("Enter the simulation parameters:")

    stimMag = float(input("stimMag: "))
    beta = float(input("beta: "))
    noise = float(input("noise: "))
    kcross = float(input("kcross: "))
    nSamples = float(input("nSamples: ")) 

    user_params = {
        'stimMag': stimMag,
        'beta': beta,   
        'noise': noise,
        'kcross': kcross,
        'nSamples': nSamples
    }

    return user_params
    
def return_simulation_info(data_path):
    user_params = read_user_input()
    match = find_file(data_path, user_params)

    if match:
        print('File found>', match)
        with h5py.File(match, 'r') as file:
            sim_id = file['id'][()][0][0].decode('utf-8')
            
    return f'simulation id: {sim_id}', match

