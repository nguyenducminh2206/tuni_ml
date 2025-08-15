# ---- LEFT PANEL (1.40) with two HDF5 modes ----
import os
import re
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import h5py
from extract_data import build_features_100samples_df
from processing_file import read_file


st.set_page_config(page_title="Simulation Dashboard", layout="wide")
st.title("📊 Simulation Dashboard")

# ---- layout: left 1.40, right 1.60  ----
left, right = st.columns([0.75, 1], gap="large")

with left:
    st.subheader("📁 Data & Configuration")

    mode = st.radio(
        "Data source", ["Upload file (≤200MB)", "Local folder (HDF5)"], index=1
    )

    df_preview = None

    if mode == "Upload file (≤200MB)":
        up = st.file_uploader("Drop file here or click to upload", type=["h5", "hdf5"])
        build1 = st.button("Build preview from uploaded file")
        if build1 and up is None:
            st.warning("Please drop a .h5/.hdf5 file first.")
        if build1 and up is not None:
            with st.spinner("Reading file…"):
                # save to a temp folder 
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, up.name if up.name.endswith(('.h5', '.hdf5')) else "uploaded.h5")
                    with open(tmp_path, "wb") as f:
                        f.write(up.getbuffer())
                    df_preview = build_features_100samples_df(tmpdir)

    else:  # folder mode
        data_folder = st.text_input("Local data folder", value="data_7x7")
        build2 = st.button("Preview data")

        if build2:
            if not os.path.isdir(data_folder):
                st.error("Folder not found.")
            else:
                # quick index info
                files = read_file(data_folder)
                if not files:
                    st.error("No .h5/.hdf5 files found.")
                else:
                    st.success(f"Found {len(files):,} files.")
                    with st.spinner("Extracting…"):
                        df_preview = build_features_100samples_df(data_folder)

    # --- show just the six columns like your screenshot ---
    if df_preview is not None and not df_preview.empty:
        keep = ["simulation_id", "sample_key", "cell_id",
                "time_trace", "dis_to_target", "cMax", "cVariance"]
        keep = [c for c in keep if c in df_preview.columns]
        df_keep = df_preview[keep].copy()

        st.session_state["features_df"] = df_keep  # for the right panel later

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("simulation_id (distinct)", df_keep["simulation_id"].nunique())
        c2.metric("sample_key (distinct)", df_keep["sample_key"].nunique())
        c3.metric("cell_id (distinct)", df_keep["cell_id"].nunique())
        c4.metric("rows", f"{len(df_keep):,}")

        st.dataframe(df_keep.head(200), use_container_width=True, height=360)

        # quick hists for numeric cols
        num_cols = [c for c in ["dis_to_target", "cMax", "cVariance"] if c in df_keep.columns]
        if num_cols:
            st.markdown("**Distributions**")
            cols = st.columns(len(num_cols))
            for name, col in zip(num_cols, cols):
                arr = df_keep[name].dropna().to_numpy()
                if arr.size > 0:
                    hist, bins = np.histogram(arr, bins=20)
                    chart_df = pd.DataFrame({"bin": (bins[:-1] + bins[1:]) / 2.0, "count": hist})
                    with col:
                        st.bar_chart(chart_df.set_index("bin"))
    elif df_preview is not None:
        st.warning("No rows extracted. Check the file/folder.")
