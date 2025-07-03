# Machine Learning BioMedical Project for Tampere University


## Overview

This project provides tools for processing, analyzing, and modeling cellwise simulation data stored in HDF5 format. The codebase includes utilities for extracting features, building machine learning datasets, and training models (MLP, CNN) to predict or classify properties such as distance to target.

## Data Structure

Simulation data is stored in HDF5 (`.h5`) files, each representing a set of simulation results with specific parameters.  
**Each file contains:**
- **`sim_ids`**: Simulation identifiers (byte strings).
- **`timeTraces`**: A group where each key (e.g., `'1'`, `'2'`, ...) corresponds to a sample. Each sample is a 2D array of shape `(n_cells, n_timepoints)` representing the time trace for each cell.
- **`tissue/distanceToTarget`**: 1D array of length `n_cells`, giving the distance to target for each cell.
- **`features`**: (optional) Group with additional features per sample.

**File naming convention:**  
Files are named to encode simulation parameters, e.g.  
`sim_data__stimMag_0.50_beta_0.40_noise_0.040_kcross_0.0050_nSamples_1000_5.h5`  
Parameters in the filename:
- `stimMag`: Stimulation magnitude
- `beta`: Beta parameter
- `noise`: Noise level
- `kcross`: Kcross parameter
- `nSamples`: Number of samples

See [`docs/hdf5_data_doc.pdf`](docs/hdf5_data_doc.pdf) for a detailed description of the HDF5 data structure.

## Main Scripts

- **`src/processing_file.py`**: Utilities for finding and parsing simulation files based on user-specified parameters.
- **`src/extract_data.py`**: Functions to extract cellwise data from HDF5 files and build pandas DataFrames for ML tasks.
- **`notebooks/mlp_model.ipynb`**: Example notebook for training an MLP regressor to predict distance to target from time traces.
- **`notebooks/cnn_classifier_model.ipynb`**: Example notebook for training a CNN classifier to categorize cells based on their time traces.

