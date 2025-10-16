# app.py
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- your local package path (keep) ---
sys.path.append(os.path.abspath('C:/Users/vpming\MyApps/tuni_ml/src'))

from pandas.api.types import is_numeric_dtype
from extract_data import build_features_10samples_df, extract_noise
from processing_file import read_file

# ML / plotting
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Page config
# =========================
st.set_page_config(page_title="Simulation Dashboard", layout="wide")
st.title("📊 Simulation Dashboard")

# =========================
# Session state defaults
# =========================
sr = st.session_state
sr.setdefault("features_df", None)           # left preview DF (with noise)
sr.setdefault("left_preview_ready", False)   # flag to render left preview after reruns
sr.setdefault("preview_bins", 40)            # left hist bins

# Right-panel persistence
sr.setdefault("model_choice", None)          # chosen model name from Section A
sr.setdefault("full_results", None)          # Section A outputs: {"acc_global", "noise_acc_df", "cm"}
sr.setdefault("drill_results", {})           # Section B outputs: {noise: {"acc", "cm"}}

# Reproducibility (optional)
np.random.seed(42)
torch.manual_seed(42)

# =========================
# Helpers (charts + profiles)
# =========================
def _trace_sig(x):
    # make arrays hashable to count distinct time_trace
    try:
        return np.asarray(x).tobytes()
    except Exception:
        return None

def _distinct_count(series, colname):
    if colname == "time_trace":
        return series.map(_trace_sig).nunique(dropna=True)
    return series.nunique(dropna=True)

def _mini_hist(series, title, bins=30, height=140):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    fig = px.histogram(
        x=s, nbins=bins, height=height,
        labels={"x": title, "y": "count"}, title=None
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), bargap=0.05)
    fig.update_xaxes(title=title, showgrid=False)
    fig.update_yaxes(title="count", showgrid=True)
    return fig

def render_wrangler(df, cols, cards_per_row=3):
    total = len(df)
    for i in range(0, len(cols), cards_per_row):
        row_cols = cols[i:i+cards_per_row]
        st_cols = st.columns(len(row_cols))
        for name, stc in zip(row_cols, st_cols):
            s = df[name]
            missing = s.isna().sum()
            miss_pct = 0 if total == 0 else round(missing/total*100)
            distinct = _distinct_count(s, name)
            dist_pct = 0 if total == 0 else round(distinct/total*100)
            with stc:
                box = st.container()
                box.markdown(f"**{name}**")
                box.markdown(
                    f"<div style='font-size:26px;font-weight:700'>{distinct:,}</div>"
                    f"<div style='opacity:.7;margin-top:-6px'>Distinct values</div>",
                    unsafe_allow_html=True
                )
                if name != "time_trace" and is_numeric_dtype(s):
                    fig = _mini_hist(s, title=name, bins=sr["preview_bins"])
                    if fig:
                        box.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================
# Models (defined once)
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, n_timepoints, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_timepoints, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

class CNNClassifier(nn.Module):
    def __init__(self, n_timepoints, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten()
        )
        lin_in = 128 * (n_timepoints // 8)
        self.fc = nn.Sequential(
            nn.Linear(lin_in, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, T]
        return self.fc(self.conv(x))

def build_model(name, n_time, n_cls):
    # keep labels consistent with your selectbox in Section A
    return CNNClassifier(n_time, n_cls) if name.startswith("Convolutional") else MLPClassifier(n_time, n_cls)

def train_simple(model, loader, epochs=15):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model

def eval_simple(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds_all.append(model(xb).argmax(dim=1).cpu().numpy())
            labels_all.append(yb.cpu().numpy())
    preds_all  = np.concatenate(preds_all) if preds_all else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])
    acc = (preds_all == labels_all).mean()*100.0 if labels_all.size else 0.0
    return acc, labels_all, preds_all

# =========================
# Layout
# =========================
left, right = st.columns([1, 1.25], gap="large")

# =========================
# LEFT panel
# =========================
with left:
    st.subheader("📁 Data & Configuration")
    mode = st.radio("Data source", ["Upload file (≤200MB)", "Local folder (HDF5)"], index=1)

    df_preview = None
    if mode == "Upload file (≤200MB)":
        up = st.file_uploader("Drop file here or click to upload", type=["h5", "hdf5"])
        build1 = st.button("Build preview from uploaded file", key="btn_preview_upload")
        if build1 and up is None:
            st.warning("Please drop a .h5/.hdf5 file first.")
        if build1 and up is not None:
            with st.spinner("Reading file…"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, up.name if up.name.endswith(('.h5', '.hdf5')) else "uploaded.h5")
                    with open(tmp_path, "wb") as f:
                        f.write(up.getbuffer())
                    df_preview = build_features_10samples_df(tmpdir)
                    # keep noise (not necessarily displayed)
                    df_preview['noise'] = df_preview['simulation_file'].apply(extract_noise)

    else:  # folder mode
        data_folder = st.text_input("Local data folder", value="data_7x7")
        build2 = st.button("Preview data", key="btn_preview_folder")
        if build2:
            if not os.path.isdir(data_folder):
                st.error("Folder not found.")
            else:
                files = read_file(data_folder)
                if not files:
                    st.error("No .h5/.hdf5 files found.")
                else:
                    st.success(f"Found {len(files):,} files.")
                    with st.spinner("Extracting…"):
                        df_preview = build_features_10samples_df(data_folder)
                        df_preview['noise'] = df_preview['simulation_file'].apply(extract_noise)

    # Save preview to state
    if df_preview is not None and not df_preview.empty:
        keep = ["simulation_id", "sample_key", "cell_id",
                "time_trace", "dis_to_target", "cMax", "cVariance", "noise"]
        keep = [c for c in keep if c in df_preview.columns]
        df_keep = df_preview[keep].copy()
        sr["features_df"] = df_keep
        sr["left_preview_ready"] = True

    # Always render from state
    df_keep = sr.get("features_df")
    if sr.get("left_preview_ready") and df_keep is not None and not df_keep.empty:
        wrangle_cols = ["simulation_id", "sample_key", "cell_id",
                        "time_trace", "cMax", "cVariance", "dis_to_target"]
        wrangle_cols = [c for c in wrangle_cols if c in df_keep.columns]
        render_wrangler(df_keep, wrangle_cols, cards_per_row=3)

        st.dataframe(
            df_keep[[c for c in ["simulation_id","sample_key","cell_id","time_trace",
                                 "dis_to_target","cMax","cVariance", "noise"] if c in df_keep.columns]].head(50),
            use_container_width=True, height=360
        )
    else:
        st.info("Load data (upload or folder) and click Preview to materialize the left-panel preview.")

# =========================
# RIGHT panel
# =========================
with right:
    st.subheader("🧪 Training & Results")

    all_df = sr.get("features_df")
    if all_df is None or all_df.empty:
        st.info("Load data on the left first.")
        st.stop()
    if "noise" not in all_df.columns:
        st.warning("Missing 'noise' column. Add it on the left before storing to session_state.")
        st.stop()

    # -------------------------
    # Section A — Balanced full dataset (single global model)
    # -------------------------
    st.markdown("### Training on balanced dataset")

    # fixed params (display only)
    EPOCHS_FULL = 15
    BATCH_FULL  = 64

    # choose model only (persist for Section B)
    model_choice = st.selectbox("Model", ["Multi-layer Perceptrons", "Convolutional Neural Network"], key="full_model")
    sr["model_choice"] = model_choice

    # show counts per dis_to_target BEFORE balancing
    per_target_counts = (
        all_df["dis_to_target"].value_counts().sort_index()
        .rename_axis("dis_to_target").reset_index(name="count")
    )
    st.markdown("**Count by dis_to_target (before balancing)**")
    st.dataframe(per_target_counts, use_container_width=True, height=180)

    min_count = int(all_df["dis_to_target"].value_counts().min())
    n_classes = int(all_df["dis_to_target"].nunique())
    st.caption(f"Balancing rule: {min_count} samples per class → total ≈ {min_count*n_classes:,} rows.")
    st.caption(f"Training params: epochs={EPOCHS_FULL}, batch_size={BATCH_FULL}")

    run_full = st.button("Training on balanced dataset", key="btn_run_full")

    if run_full:
        # Balance by dis_to_target (whole dataset)
        df_use  = all_df.copy()
        min_cnt = int(df_use["dis_to_target"].value_counts().min())
        df_bal  = (df_use.groupby("dis_to_target", group_keys=False)
                        .apply(lambda x: x.sample(n=min_cnt, random_state=42))
                        .reset_index(drop=True))

        # Split/scale
        X      = np.stack(df_bal["time_trace"].values)
        y      = df_bal["dis_to_target"].astype(int).values
        nz_vec = df_bal["noise"].values

        strat = y if len(np.unique(y)) > 1 else None
        Xtr, Xte, ytr, yte, nz_tr, nz_te = train_test_split(
            X, y, nz_vec, test_size=0.2, random_state=42, stratify=strat
        )
        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(Xtr)
        Xte_s  = scaler.transform(Xte)

        tr_ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
        te_ds = TensorDataset(torch.tensor(Xte_s, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_FULL, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_FULL)

        n_time = Xtr.shape[1]; n_cls = int(np.max(ytr)) + 1
        model  = build_model(model_choice, n_time, n_cls)

        with st.spinner("Training…"):
            model = train_simple(model, tr_ld, epochs=EPOCHS_FULL)

        acc_global, labels_g, preds_g = eval_simple(model, te_ld)
        st.markdown(f"**Test accuracy:** `{acc_global:.2f}%`")

        # Accuracy vs Noise (computed on test split of the global model)
        per_noise = []
        for nz in sorted(np.unique(nz_te)):
            m = (nz_te == nz)
            if m.sum() == 0: continue
            X_slice = torch.tensor(Xte_s[m], dtype=torch.float32)
            with torch.no_grad():
                pred = model(X_slice).argmax(dim=1).cpu().numpy()
            acc_nz = (pred == yte[m]).mean() * 100.0
            per_noise.append({"noise": float(nz), "acc": float(acc_nz)})

        res_df = pd.DataFrame(per_noise).sort_values("noise") if per_noise else pd.DataFrame(columns=["noise","acc"])

        # Global confusion matrix
        cm_g = confusion_matrix(labels_g, preds_g) if labels_g.size and preds_g.size else np.zeros((n_classes, n_classes), dtype=int)

        # Persist Section A results so they don't disappear
        sr["full_results"] = {
            "acc_global": float(acc_global),
            "noise_acc_df": res_df.copy(),
            "cm": cm_g,
        }

    # Always render last Section A results
    saved = sr.get("full_results")
    if saved:
        st.markdown(f"**Test accuracy:** `{saved['acc_global']:.2f}%`")
        if not saved["noise_acc_df"].empty:
            fig_line = px.line(
                saved["noise_acc_df"].sort_values("noise"),
                x="noise", y="acc", markers=True,
                labels={"noise":"Noise", "acc":"Accuracy (%)"},
                title="Accuracy vs Noise"
            )
            st.plotly_chart(fig_line, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        sns.heatmap(saved["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    st.divider()

    # -------------------------
    # Section B — Drilldown by a single noise
    # -------------------------
    st.markdown("### Training on single noise")

    # fixed params for drilldown (display only)
    EPOCHS_NOISE = 10
    BATCH_NOISE  = 64

    chosen_model = sr.get("model_choice", "Multi-layer Perceptrons")
    st.caption(f"Using model: {chosen_model} • epochs={EPOCHS_NOISE}, batch_size={BATCH_NOISE}")

    noise_opts = sorted(all_df["noise"].dropna().unique())
    pick_noise = st.selectbox("Noise", noise_opts, index=0, format_func=lambda x: f"{x:g}")
    run_noise  = st.button("Train", key="btn_run_noise")

    if run_noise:
        df_n = all_df[all_df["noise"] == pick_noise].copy()
        if df_n.empty:
            st.warning("No data for this noise.")
        else:
            # Balance within this noise
            min_cnt_n = int(df_n["dis_to_target"].value_counts().min())
            df_bal_n  = (df_n.groupby("dis_to_target", group_keys=False)
                            .apply(lambda x: x.sample(n=min_cnt_n, random_state=42))
                            .reset_index(drop=True))

            # Split/scale
            X = np.stack(df_bal_n["time_trace"].values)
            y = df_bal_n["dis_to_target"].astype(int).values
            strat = y if len(np.unique(y)) > 1 else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
            scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)

            n_time = Xtr.shape[1]; n_cls = int(np.max(ytr)) + 1
            tr_ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
            te_ds = TensorDataset(torch.tensor(Xte_s, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
            tr_ld = DataLoader(tr_ds, batch_size=BATCH_NOISE, shuffle=True)
            te_ld = DataLoader(te_ds, batch_size=BATCH_NOISE)

            # Use model chosen in Section A
            model_n = build_model(chosen_model, n_time, n_cls)
            with st.spinner(f"Training on noise={pick_noise:g}…"):
                model_n = train_simple(model_n, tr_ld, epochs=EPOCHS_NOISE)

            acc_n, labels_n, preds_n = eval_simple(model_n, te_ld)
            cm_n = confusion_matrix(labels_n, preds_n)

            # Persist per-noise result and render
            sr["drill_results"][float(pick_noise)] = {"acc": float(acc_n), "cm": cm_n}

    # Always render the last run for currently selected noise (if any)
    saved_n = sr["drill_results"].get(float(pick_noise))
    if saved_n:
        st.markdown(f"**Noise {pick_noise:g} — test accuracy:** `{saved_n['acc']:.2f}%`")
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        sns.heatmap(saved_n["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (noise={pick_noise:g})")
        st.pyplot(fig)
