import os
import tempfile
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
sys.path.append(os.path.abspath('C:/Users/vpming/tuni_ml/src'))
from pandas.api.types import is_numeric_dtype
from extract_data import build_features_100samples_df
from processing_file import read_file


st.set_page_config(page_title="Simulation Dashboard", layout="wide")
st.title("📊 Simulation Dashboard")

# ---- layout ----
left, right = st.columns([1.25, 1], gap="large")

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

    # --- show six columns ---
    if df_preview is not None and not df_preview.empty:
        keep = ["simulation_id", "sample_key", "cell_id",
                "time_trace", "dis_to_target", "cMax", "cVariance"]
        keep = [c for c in keep if c in df_preview.columns]
        df_keep = df_preview[keep].copy()

        st.session_state["features_df"] = df_keep  # for the right panel later

        # --- Data Wrangler–style summary cards ---
        def _trace_sig(x):
            # make arrays hashable for distinct counting
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
            fig = px.histogram(x=s, nbins=bins, height=height,
                            labels={"x": title, "y": "count"},
                            title=None)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), bargap=0.05)
            fig.update_xaxes(title=title, showgrid=False)
            fig.update_yaxes(title="count", showgrid=True)
            return fig

        def render_wrangler(df, cols, cards_per_row=3):
            total = len(df)
            # chunk columns into rows
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
                        box.caption(f"Missing: {missing} ({miss_pct}%)  •  Distinct: {distinct} ({dist_pct}%)")
                        # big distinct number
                        box.markdown(
                            f"<div style='font-size:26px;font-weight:700'>{distinct:,}</div>"
                            f"<div style='opacity:.7;margin-top:-6px'>Distinct values</div>",
                            unsafe_allow_html=True
                        )
                        # chart (numeric only; we don't try to histogram time_trace)
                        if name != "time_trace" and is_numeric_dtype(s):
                            fig = _mini_hist(s, title=name)
                            if fig:
                                box.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # choose the exact columns/order
        wrangle_cols = ["simulation_id", "sample_key", "cell_id",
                        "time_trace", "cMax", "cVariance", "dis_to_target"]

        # only include columns that exist
        wrangle_cols = [c for c in wrangle_cols if c in df_keep.columns]

        render_wrangler(df_keep, wrangle_cols, cards_per_row=3)

        # Table preview (like Data Wrangler shows the rows below the profile)
        st.dataframe(df_keep.head(200), use_container_width=True, height=360)
