# Machine Learning BioMedical Project (Tampere University)

A streamlined toolkit to **ingest, profile, and model** cell-wise simulation data stored in **HDF5**. It includes:

- Fast **data extraction** into tidy DataFrames (per cell, per sample).
- A **Streamlit** app for interactive data wrangling, model training, and drill-downs.
- Ready-to-use **MLP** and **1D CNN** classifiers for cell classification tasks.

---

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && . .venv/Scripts/activate   # Windows
# or: python -m venv .venv && source .venv/bin/activate  # macOS/Linux

# 2) Install deps
pip install -U pip
pip install streamlit numpy pandas h5py torch scikit-learn seaborn matplotlib plotly

# 3) Run the app (from repo root)
streamlit run app.py
```

## Data format (HDF5)

Each `.h5` file represents one simulation batch and typically encodes parameters in the filename, e.g.  
`sim_data__stimMag_0.50_beta_0.40_noise_0.040_kcross_0.0050_nSamples_1000_5.h5`

Inside each file:

- **`sim_ids`** — list of simulation identifiers (bytes).
- **`timeTraces/<sample_key>`** — 2D array `[n_cells, n_timepoints]` time-series per cell.
- **`tissue/distanceToTarget`** — 1D array `[n_cells]` target distance per cell.
- **`features/*`** *(optional)* — additional per-sample feature vectors (e.g., `cMax`, `cVariance`, …).

Filename parameters commonly include: `stimMag`, `beta`, `noise`, `kcross`, `nSamples`.

> For a deeper spec, see `docs/hdf5_data_doc.pdf` (if present in your repo).

---

## Project structure

```
.
├─ app.py                        # Streamlit UI (ingest, wrangle, train)
├─ src/
│  ├─ extract_data.py            # build_features_10samples_df, extract_noise, …
│  └─ processing_file.py         # read_file (find .h5 files recursively)
├─ docs/
│  └─ hdf5_data_doc.pdf          # optional: data format doc
└─ notebooks/                    # optional: experiments
```

---

## Streamlit app

### Left panel — Data & Configuration
Two ways to load data (**HDF5 only**):

1. **Upload** a single `.h5/.hdf5` (saved to a temp folder behind the scenes).  
2. **Folder path** to a directory containing multiple `.h5` files (recursively scanned).

The app extracts a compact, cell-wise preview (first *N* samples/file in `extract_data.py`; default **10**) and shows a **wrangler‑style summary** for:

- `simulation_id`, `sample_key`, `cell_id`, `time_trace`,  
- `dis_to_target`, `cMax`, `cVariance`, `noise` (parsed from filename).



### Right panel — Training & Results

#### Section A — Balanced full dataset (single global model)
- **Balance rule:** compute `min_count = value_counts(dis_to_target).min()` and sample `min_count` rows **per `dis_to_target`** across **all noises**.
- Train one model (choose **Multi-layer Perceptrons** or **Convolutional Neural Network**).  
  Fixed params (display only): `epochs=15`, `batch_size=64`.
- **Outputs:**
  - **Accuracy vs Noise** — evaluates the global model on the test split, grouped by noise.
  - **Global confusion matrix** — on the held‑out test split.

#### Section B — Drilldown by single noise
- Choose one **noise** value (e.g., `0.01`, `0.02`, …).
- Balance **within** that noise over `dis_to_target`, split, scale, and train using the **same model type** picked in Section A.  
  Fixed params (display only): `epochs=10`, `batch_size=64`.
- **Outputs:** noise‑specific **test accuracy** and **confusion matrix**.

All results (plots, matrices, chosen model) are persisted to `st.session_state` so they remain visible across app interactions.

---

## Key modules

- **`src/extract_data.py`**
  - `build_features_10samples_df(data_path)` — walks `.h5` files and builds a cell‑wise DataFrame with:  
    `simulation_id`, `sample_key`, `cell_id`, `time_trace`, `dis_to_target`, `cMax`, `cVariance`, `simulation_file`.
  - `extract_noise(filename)` — parses the `noise` level from the filename.
- **`src/processing_file.py`**
  - `read_file(folder_path)` — recursively collect `.h5`/`.hdf5` paths.

---

## Modeling details

- **Input `X`** — stacked `time_trace` arrays → shape `[n_samples, n_timepoints]`.
- **Target `y`** — discrete `dis_to_target` labels (balanced over exact values).  
  *(If your data is continuous, adapt by binning before balancing.)*
- **Scaling** — `StandardScaler` on **X** only.
- **Architectures**  
  - **MLP**: Linear(→256)–ReLU–Dropout–Linear(→128)–ReLU–Dropout–Linear(→C).  
  - **1D CNN**: Conv1d(1→32,k=7) → MaxPool → Conv1d(32→64,k=5) → MaxPool → Conv1d(64→128,k=3) → MaxPool → Flatten → same FC head.
- **Metrics** — Test accuracy, confusion matrix; per‑noise accuracy derived from the global model’s test split (Section A).

---

## Example workflow

1. Run the app: `streamlit run app.py`  
2. **Left panel**: choose *Upload* or *Folder* → click **Preview** → review summary + table.  
3. **Right / Section A**: pick **model** → click **Training on balanced dataset** → view **Accuracy vs Noise** + **Confusion Matrix**.  
4. **Right / Section B**: pick a **noise** → click **Train** → view noise‑specific **accuracy** + **confusion matrix**.



