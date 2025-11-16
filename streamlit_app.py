
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_PATH = "rf_terror_model_full_tuned_shap.joblib"
DATA_PATH = "globalterrorismdb_2021Jan-June_1222dist.xlsx"

st.set_page_config(page_title="GTD Prediction Dashboard", layout="wide")

st.title("💣 Global Terrorism Analysis & Fatality Prediction Dashboard")
st.write("Explore historical attack trends and predict fatality likelihood using a trained ML model.")

# -------------------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load dataset
try:
    df = load_data()
    st.success(f"Dataset loaded successfully: {df.shape[0]} rows")
except:
    st.error("❌ Could not load dataset. Ensure the Excel file is in the repository root.")
    df = None

# Load trained model
try:
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    st.success("Model loaded successfully.")
except:
    st.error("❌ Could not load model file. Ensure the .joblib file is in the repository root.")
    st.stop()

# -------------------------------------------------------
# DATA VISUALIZATIONS
# -------------------------------------------------------
st.header("📊 Data Visualizations")

if df is not None:
    tab1, tab2 = st.tabs(["Attack Types Distribution", "Fatalities by Region"])

    # Pie chart for attack types
    with tab1:
        if "attacktype1_txt" in df.columns:
            fig = px.pie(
                df,
                names="attacktype1_txt",
                title="Distribution of Attack Types",
                hole=0.4,
                width=700,
                height=450
            )
            st.plotly_chart(fig, width="stretch")

    # Bar chart for fatalities
    with tab2:
        if {"region_txt", "nkill"}.issubset(df.columns):
            regional = df.groupby("region_txt")["nkill"].sum().reset_index()
            fig = px.bar(
                regional,
                x="region_txt",
                y="nkill",
                title="Total Fatalities by Region",
                width=900,
                height=500,
                color="nkill",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------
# FATALITY PREDICTION
# -------------------------------------------------------
st.header("🎯 Fatality Prediction")

num_features = ["latitude", "longitude", "month", "day"]

col1, col2 = st.columns(2)

latitude = col1.number_input("Latitude", -90.0, 90.0, 20.0)
longitude = col2.number_input("Longitude", -180.0, 180.0, 80.0)
month = col1.slider("Month", 1, 12, 6)
day = col2.slider("Day", 1, 31, 15)

sample = pd.DataFrame({
    "latitude": [latitude],
    "longitude": [longitude],
    "month": [month],
    "day": [day]
})

# Add categorical features based on preprocessor
cat_features = []
for name, transformer, cols in preprocessor.transformers_:
    if name == "cat":
        cat_features = cols

if df is not None:
    for cat in cat_features:
        if cat in df.columns:
            options = df[cat].dropna().unique().tolist()
            val = st.selectbox(f"{cat}", options)
            sample[cat] = [val]
        else:
            sample[cat] = ["Other"]

# Make prediction
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

if pred == 1:
    st.error(f"☠️ Predicted: Fatal Attack (Probability: {prob:.2f})")
else:
    st.success(f"🕊️ Predicted: Non-Fatal Attack (Probability: {prob:.2f})")

# -------------------------------------------------------
# SHAP FEATURE IMPORTANCE
# -------------------------------------------------------
st.header("🔍 SHAP Feature Importance")

explainer = shap.TreeExplainer(classifier)

try:
    sample_trans = preprocessor.transform(sample)
    shap_values = explainer.shap_values(sample_trans)

    # Draw SHAP bar plot safely
    st.subheader("Top SHAP Feature Contributions")
    plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values[1], sample_trans, plot_type="bar", show=False)
    st.pyplot(plt)
except Exception as e:
    st.warning(f"SHAP could not generate explanations: {e}")

st.caption("Model trained on GTD dataset | Random Forest + SHAP explainability | Streamlit Dashboard by Ananya Pandey")
