import streamlit as st
import pandas as pd
import joblib
import json


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

  from xgboost import XGBRegressor
    model = XGBRegressor() 
      model.load_model("model/xgboost_model.json")




    feature_columns = joblib.load(
        "model/feature_columns.pkl"
    )

    historical_data = joblib.load(
        "model/historical_data.pkl"
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    with open("model/metadata.json", "r") as f:
        metadata = json.load(f)

    return (
        model,
        feature_columns,
        historical_data,
        scaler,
        metadata
    )


# ============================================================
# LOAD
# ============================================================

try:

    (
        model,
        feature_columns,
        historical_data,
        scaler,
        metadata
    ) = load_model()

    st.success("XGBoost model loaded successfully!")

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📈 Sales Forecasting Dashboard")

st.write(
    "XGBoost Product × Store Sales Forecasting"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "XGBoost"
    )

with col2:

    st.metric(
        "Features",
        len(feature_columns)
    )

with col3:

    st.metric(
        "Forecast Horizon",
        "30 Days"
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

st.header("Historical Data")

st.write(
    f"Historical rows: {len(historical_data):,}"
)

st.write(
    f"Products: {historical_data['Product_ID'].nunique()}"
)

st.write(
    f"Stores: {historical_data['Store_ID'].nunique()}"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

metrics = metadata.get(
    "validation",
    {}
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "MAE",
        f"{metrics.get('MAE', 0):.2f}"
    )

with col2:

    st.metric(
        "RMSE",
        f"{metrics.get('RMSE', 0):.2f}"
    )

with col3:

    st.metric(
        "WMAPE",
        f"{metrics.get('WMAPE', 0):.2f}%"
    )

with col4:

    st.metric(
        "WMAPE Accuracy",
        f"{metrics.get('WMAPE_Accuracy', 0):.2f}%"
    )


# ============================================================
# PRODUCT / STORE SELECTION
# ============================================================

st.header("Sales Forecast")

product_list = sorted(
    historical_data["Product_ID"]
    .dropna()
    .unique()
)

store_list = sorted(
    historical_data["Store_ID"]
    .dropna()
    .unique()
)

col1, col2 = st.columns(2)

with col1:

    product = st.selectbox(
        "Select Product",
        product_list
    )

with col2:

    store = st.selectbox(
        "Select Store",
        store_list
    )


# ============================================================
# SHOW HISTORY
# ============================================================

selected_history = historical_data[
    (historical_data["Product_ID"] == product) &
    (historical_data["Store_ID"] == store)
].copy()

st.subheader(
    f"Historical Sales — {product} / {store}"
)

st.dataframe(
    selected_history[
        [
            "Date",
            "Product_ID",
            "Store_ID",
            "Units_Sold"
        ]
    ].sort_values("Date"),
    use_container_width=True
)


# ============================================================
# STATUS
# ============================================================

st.info(
    "The model and historical data are successfully connected. "
    "The next step is connecting the exact corrected forecasting "
    "feature-engineering pipeline to generate future predictions."
)
