# -*- coding: utf-8 -*-
"""RWAP Task 2 Analytical Dashboard"""

import os
import re
import io
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional spatial analysis
try:
    from esda.moran import Moran
    from libpysal.weights import KNN
    PYSAL_AVAILABLE = True
except Exception:
    PYSAL_AVAILABLE = False

st.set_page_config(page_title="RWAP Task 2 – Analytical Dashboard", layout="wide")

# --------------------------
# Utilities
# --------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes_or_path):
    """Load CSV from uploaded file or local path."""
    if isinstance(file_bytes_or_path, (str, os.PathLike)):
        return pd.read_csv(file_bytes_or_path, low_memory=False)
    else:
        return pd.read_csv(file_bytes_or_path, low_memory=False)

def detect_date_cols(df):
    pat = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    date_cols = [c for c in df.columns if pat.match(str(c))]
    date_cols_sorted = sorted(date_cols, key=lambda x: pd.to_datetime(x, dayfirst=True))
    return date_cols_sorted

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def kpi_box(label, value, helptext=None):
    st.metric(label, value if value is not None else "—", help=helptext)

def prep_categoricals_for_catboost(df, cat_cols):
    X = df.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "Unknown").astype(str)
        X.loc[X[c].isin(["nan","NaN","None","<NA>","NaT"]), c] = "Unknown"
        X[c] = X[c].str.strip()
    return X

def compute_moran_i(df, value_col, lat_col="Latitude", lon_col="Longitude", k=8):
    """KNN-based Moran's I (no shapefile required). Returns (I, p_value) or (None, None)."""
    try:
        pts = df[[lat_col, lon_col, value_col]].dropna()
        if len(pts) < k + 3:
            return None, None
        coords = pts[[lat_col, lon_col]].values
        w = KNN.from_array(coords, k=k)
        w.transform = "R"  # row-standardized
        y = pts[value_col].values
        mi = Moran(y, w)
        return float(mi.I), float(mi.p_sim)
    except Exception:
        return None, None

def format_money(x):
    try:
        if pd.isna(x): return "—"
        if abs(x) >= 1e9: return f"{x/1e9:.2f} B"
        if abs(x) >= 1e6: return f"{x/1e6:.2f} M"
        if abs(x) >= 1e3: return f"{x/1e3:.2f} K"
        return f"{x:.0f}"
    except Exception:
        return "—"

# --------------------------
# Sidebar – Data & Model
# --------------------------
st.sidebar.header("Data & Model")

sample_data_note = (
    "Tip: Use the enriched file from Task 1 (e.g., "
    "`task1_valuations_timeaware.csv` or `task1_valuations_with_preds.csv`)."
)
st.sidebar.info(sample_data_note)

# File upload
df = None
up = st.sidebar.file_uploader("Upload dataset (.csv)", type=["csv"])
if up is not None:
    df = pd.read_csv(up, low_memory=False)

# --------------------------
# Apply JSON mapping to decode encoded columns
# --------------------------
json_file = st.sidebar.file_uploader("Upload combined JSON mapping (.json)", type=["json"])
mappings = {}
if json_file is not None:
    raw_map = json.load(io.TextIOWrapper(json_file, encoding="utf-8"))
    # Invert mapping: code -> label
    mappings = {}
    for col, mapping in raw_map.items():
        inverted = {str(code): label for label, code in mapping.items()}
        mappings[col] = inverted
        # Also handle suffixed columns like "State_d1"
        mappings[col + "_d1"] = inverted





def decode_columns(df, mappings):
    df_decoded = df.copy()
    for col, mapping in mappings.items():
        if col in df_decoded.columns:
            # Convert everything to string for safe mapping
            df_decoded[col] = df_decoded[col].astype(str).map(lambda x: mapping.get(x, x))
    return df_decoded

df = decode_columns(df, mappings)

if df is not None and json_file is not None:
    df = decode_columns(df, mappings)
    st.write("Decoded columns preview:")
    st.dataframe(df.head(10))


# Model loader (optional)
st.sidebar.subheader("Valuation Model (optional)")
model_path = st.sidebar.text_input("CatBoost model path (.cbm or .joblib)", value="asset_valuation_model.cbm")
loaded_model = None
catboost_loaded = False
cat_cols_for_model = ["Owned or Leased","GSA Region","State_d1","City_d1",
                      "Real Property Asset Type","Building Status","Metro","CountyName"]
num_cols_for_model = ["Building Rentable Square Feet","Building Age","Latitude","Longitude","valuation_per_sqft"]

if model_path and os.path.exists(model_path):
    try:
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor()
        model.load_model(model_path)
        loaded_model = model
        catboost_loaded = True
        st.sidebar.success("CatBoost model loaded")
    except Exception:
        try:
            loaded_model = joblib.load(model_path)
            st.sidebar.success("Model loaded (joblib)")
        except Exception as e:
            st.sidebar.error(f"Could not load model: {e}")

# --------------------------
# Main
# --------------------------
st.title("RWAP 2025–26 — Task 2 Analytical Dashboard")
st.caption("Descriptive & Inferential stats • Time-series explorer • Mapping & Spatial analysis")

if df is None:
    st.warning("Upload or load a CSV to begin.")
    st.stop()

# Basic hygiene
date_cols_sorted = detect_date_cols(df)
has_geo = ("Latitude" in df.columns) and ("Longitude" in df.columns)

# Choose valuation column
val_options = []
if "predicted_valuation" in df.columns:
    val_options.append("predicted_valuation")
if "valuation_proxy" in df.columns:
    val_options.append("valuation_proxy")
chosen_val_col = st.selectbox("Valuation column to analyze", val_options or df.columns.tolist())

# --------------------------
# Filters
# --------------------------
with st.expander("Filters", expanded=True):
    cols_left, cols_right = st.columns(2)
    with cols_left:
        state_sel = st.multiselect("State", sorted(df["State_d1"].dropna().astype(str).unique()) if "State_d1" in df.columns else [])
        city_sel  = st.multiselect("City",  sorted(df["City_d1"].dropna().astype(str).unique()) if "City_d1" in df.columns else [])
        type_sel  = st.multiselect("Asset Type (code)", sorted(df["Real Property Asset Type"].dropna().unique()) if "Real Property Asset Type" in df.columns else [])
    with cols_right:
        own_sel   = st.multiselect("Owned or Leased", sorted(df["Owned or Leased"].dropna().astype(str).unique()) if "Owned or Leased" in df.columns else [])
        status_sel= st.multiselect("Building Status", sorted(df["Building Status"].dropna().astype(str).unique()) if "Building Status" in df.columns else [])
        zip_sel   = st.multiselect("Zip Code", sorted(df["Zip Code"].dropna().astype(str).unique()) if "Zip Code" in df.columns else [])

# Apply filters
mask = pd.Series(True, index=df.index)
if state_sel and "State_d1" in df.columns:  mask &= df["State_d1"].astype(str).isin(state_sel)
if city_sel  and "City_d1" in df.columns:   mask &= df["City_d1"].astype(str).isin(city_sel)
if type_sel  and "Real Property Asset Type" in df.columns: mask &= df["Real Property Asset Type"].isin(type_sel)
if own_sel   and "Owned or Leased" in df.columns: mask &= df["Owned or Leased"].astype(str).isin(own_sel)
if status_sel and "Building Status" in df.columns: mask &= df["Building Status"].astype(str).isin(status_sel)
if zip_sel and "Zip Code" in df.columns: mask &= df["Zip Code"].astype(str).isin(zip_sel)

dff = df[mask].copy()
st.write(f"**Filtered rows:** {len(dff):,} of {len(df):,}")

# --------------------------
# KPIs
# --------------------------
k1, k2, k3, k4 = st.columns(4)
kpi_box("Median Valuation", format_money(dff[chosen_val_col].median() if chosen_val_col in dff else None))
kpi_box("Average Valuation", format_money(dff[chosen_val_col].mean() if chosen_val_col in dff else None))
kpi_box("Asset Types (distinct)", dff["Real Property Asset Type"].nunique() if "Real Property Asset Type" in dff.columns else None)
kpi_box("Geocoded Assets", int(dff["Latitude"].notna().sum()) if has_geo else None)

st.divider()

# --------------------------
# Tabs: Overview • Time Series • Map • Spatial • Segments • Data
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Time series explorer", "Map (GIS)", "Spatial analysis", "Segments", "Data & Download"]
)

# ---- Overview
with tab1:
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Distribution of valuation")
        st.bar_chart(dff[chosen_val_col].dropna().clip(upper=dff[chosen_val_col].quantile(0.99)))
    with c2:
        st.subheader("By Asset Type (median)")
        if "Real Property Asset Type" in dff.columns:
            by_type = dff.groupby("Real Property Asset Type")[chosen_val_col].median().sort_values(ascending=False)
            st.dataframe(by_type.reset_index().rename(columns={chosen_val_col:"Median Valuation"}))
        else:
            st.info("Column `Real Property Asset Type` not found.")
    st.subheader("Top 15 assets by valuation")
    show_cols = [c for c in ["Street Address","City_d1","State_d1","Zip Code","Real Property Asset Type", chosen_val_col] if c in dff.columns]
    st.dataframe(dff.nlargest(15, chosen_val_col)[show_cols])

# ---- Time series explorer
with tab2:
    st.subheader("Index time series (2000–2025)")
    if not date_cols_sorted:
        st.info("No monthly index columns found.")
    else:
        level = st.selectbox("Group by", ["(none)","State_d1","City_d1","Zip Code"])
        if level == "(none)":
            ts = dff[date_cols_sorted].mean(axis=0, skipna=True)
            st.line_chart(ts)
        else:
            if level not in dff.columns:
                st.warning(f"{level} not in dataset.")
            else:
                top_n = st.slider("Top groups (by row count)", 3, 15, 5)
                groups = dff[level].astype(str).value_counts().head(top_n).index.tolist()
                ts_df = dff[dff[level].astype(str).isin(groups)]
                m = ts_df.groupby(level)[date_cols_sorted].mean()
                st.line_chart(m.T)

# ---- Map (GIS)
with tab3:
    st.subheader("Geospatial view")
    if has_geo:
        sample_n = st.slider("Sample points (for performance)", 1000, 10000, 3000, step=500)
        dmap = dff[["Latitude","Longitude", chosen_val_col]].dropna().sample(min(sample_n, len(dff)), random_state=42)
        st.map(dmap.rename(columns={"Latitude":"lat","Longitude":"lon"}))
        st.caption("Tip: Zoom and pan. Color scale is uniform; use filters to focus.")
    else:
        st.info("No Latitude/Longitude columns; cannot render map.")

# ---- Spatial analysis (Moran's I with KNN)
with tab4:
    st.subheader("Spatial autocorrelation (Moran's I)")
    if not has_geo:
        st.info("No coordinates available.")
    else:
        if not PYSAL_AVAILABLE:
            st.warning("Install `esda` and `libpysal` to enable Moran's I:  \n`pip install esda libpysal`")
        else:
            k = st.slider("KNN neighbors (k)", 4, 20, 8)
            I, p = compute_moran_i(dff, chosen_val_col, "Latitude", "Longitude", k=k)
            if I is None:
                st.info("Not enough points for this k.")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Moran's I", f"{I:.3f}", help=">0: clustering, <0: dispersion, ~0: random")
                col2.metric("p-value (permutation)", f"{p:.4f}", help="Small p suggests significant spatial structure")

# ---- Segments
with tab5:
    st.subheader("Segment summaries")
    seg1, seg2 = st.columns(2)
    if "State_d1" in dff.columns:
        with seg1:
            st.write("By State (median)")
            s1 = dff.groupby("State_d1")[chosen_val_col].median().sort_values(ascending=False).head(20)
            st.dataframe(s1.reset_index().rename(columns={chosen_val_col:"Median valuation"}))
    if "City_d1" in dff.columns:
        with seg2:
            st.write("By City (median)")
            s2 = dff.groupby("City_d1")[chosen_val_col].median().sort_values(ascending=False).head(20)
            st.dataframe(s2.reset_index().rename(columns={chosen_val_col:"Median valuation"}))

# ---- Data & Download
with tab6:
    st.subheader("Filtered data preview")
    st.dataframe(dff.head(200))
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_data.csv")

    # Batch prediction for currently filtered rows (if model provided)
    st.markdown("---")
    st.subheader("Predict valuation for filtered rows (optional)")
    if loaded_model is None:
        st.info("Load a CatBoost/joblib model in the sidebar to enable prediction.")
    else:
        cat_cols_present = [c for c in cat_cols_for_model if c in dff.columns]
        num_cols_present = [c for c in num_cols_for_model if c in dff.columns]
        feat_cols = cat_cols_present + num_cols_present

        X_full = dff[feat_cols].copy()
        X_full = prep_categoricals_for_catboost(X_full, cat_cols_present)
        for c in num_cols_present:
            X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
            X_full[c] = X_full[c].fillna(X_full[c].median())

        try:
            from catboost import Pool
            cat_idx = [X_full.columns.get_loc(c) for c in cat_cols_present]
            preds = loaded_model.predict(Pool(X_full, cat_features=cat_idx if len(cat_idx)>0 else None))
        except Exception:
            preds = loaded_model.predict(X_full)

        dff_out = dff.copy()
        dff_out["predicted_valuation_model"] = preds
        st.dataframe(dff_out.head(200))

        csvp = dff_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download with predictions", csvp, file_name="filtered_with_predictions.csv")
