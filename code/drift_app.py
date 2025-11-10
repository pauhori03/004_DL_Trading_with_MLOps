"""
drift_app.py (Streamlit)
------------------------
Simple UI to visualize drift results. It loads train/val/test splits and the
precomputed drift table, lets the user pick a feature, and shows distributions.
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("Feature Drift Dashboard")

# Load splits
splits_dir = "data/splits"
drift_path = "data/processed/drift_table.csv"

if not os.path.exists(splits_dir):
    st.error("No splits found. Run `python main.py` first.")
    st.stop()

X_train = pd.read_csv(os.path.join(splits_dir, "X_train.csv"), index_col=0, parse_dates=True)
X_val = pd.read_csv(os.path.join(splits_dir, "X_val.csv"), index_col=0, parse_dates=True)
X_test = pd.read_csv(os.path.join(splits_dir, "X_test.csv"), index_col=0, parse_dates=True)

if os.path.exists(drift_path):
    drift_tbl = pd.read_csv(drift_path)
else:
    st.warning("No drift table found. It will be created after pipeline run.")
    drift_tbl = pd.DataFrame()

if not drift_tbl.empty:
    st.subheader("Drift Table (sorted by Train vs Test p-value)")
    st.dataframe(drift_tbl)

    # Top-5 drifted features
    top5 = drift_tbl.sort_values("p_train_vs_test").head(5)
    st.markdown("### Top-5 Most Drifted Features (lowest p-values)")
    for _, row in top5.iterrows():
        st.write(f"- **{row['feature']}** | p_train_vs_test={row['p_train_vs_test']:.4f} | drift_test={row['drift_test']}")
else:
    st.info("Drift table not available to display.")

# Feature selection
st.subheader("Distribution Viewer")
feature = st.selectbox("Select a feature", options=X_train.columns.tolist())

if feature:
    fig1 = plt.figure()
    X_train[feature].dropna().hist(bins=50)
    plt.title(f"Train Distribution: {feature}")
    st.pyplot(fig1)

    fig2 = plt.figure()
    X_val[feature].dropna().hist(bins=50)
    plt.title(f"Val Distribution: {feature}")
    st.pyplot(fig2)

    fig3 = plt.figure()
    X_test[feature].dropna().hist(bins=50)
    plt.title(f"Test Distribution: {feature}")
    st.pyplot(fig3)
