import os
import re
import h5py

def read_file(folder_path):
    """
    Find all .h5 files 
    """
    h5_files = []
    files = os.listdir(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

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
    for file_path in files:
        file_name = os.path.basename(file_path)
        params = parse_simulation(file_name)
        if params and all(abs(params[key] - user_params[key]) < 1e-6 for key in user_params):
            return file_path
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
            file_name = os.path.basename(match)
            sim_id = file['id'][()][0][0].decode('utf-8')
            print(f'File name:{file_name}')
            return f'simulation id: {sim_id}', match

    else:
        return 'File not found'

def main():
    data_path = 'data'
    print(read_file(data_path))

if __name__ == "__main__":
    main()