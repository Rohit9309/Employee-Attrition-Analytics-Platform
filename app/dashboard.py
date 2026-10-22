
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HR_Attrition_Indian_Dataset.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Employee Attrition Analytics", page_icon="📊", layout="wide")
st.title("Employee Attrition Analytics Platform")
st.caption("Educational HR analytics prototype — use predictions as decision support only.")

df = pd.read_csv(DATA_PATH)
if "LeftCompany" not in df.columns:
    st.error("Dataset must include LeftCompany.")
    st.stop()
df["LeftCompany"] = df["LeftCompany"].astype(str)
total = len(df)
left = int((df["LeftCompany"] == "Yes").sum())
rate = left / total if total else 0
retention = 1 - rate
c1,c2,c3,c4 = st.columns(4)
c1.metric("Employees in dataset", f"{total:,}")
c2.metric("Attrition rate", f"{rate:.1%}")
c3.metric("Retention rate", f"{retention:.1%}")
c4.metric("Departments", str(df["Department"].nunique()) if "Department" in df else "N/A")

st.divider()
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Attrition by department")
    if "Department" in df.columns:
        dep = df.groupby("Department", dropna=False)["LeftCompany"].apply(lambda s: (s=="Yes").mean()).mul(100).reset_index(name="Attrition rate (%)").sort_values("Attrition rate (%)", ascending=False)
        st.plotly_chart(px.bar(dep, x="Department", y="Attrition rate (%)", title="Observed attrition rate by department"), use_container_width=True)
    else:
        st.info("Department column not found.")
with right_col:
    st.subheader("Attrition outcome")
    st.plotly_chart(px.pie(df, names="LeftCompany", title="Stayed vs left"), use_container_width=True)

st.subheader("Workforce demographics")
demographic_cols = [c for c in ["Age", "Gender", "Department", "JobRole", "Education", "MaritalStatus", "WorkMode"] if c in df.columns]
if demographic_cols:
    selected = st.selectbox("Choose a field", demographic_cols)
    counts = df[selected].fillna("Missing").astype(str).value_counts().rename_axis(selected).reset_index(name="Employees")
    st.plotly_chart(px.bar(counts, x=selected, y="Employees", title=f"Employee distribution by {selected}"), use_container_width=True)

st.divider()
st.subheader("Employee attrition prediction")
st.write("Enter employee attributes below. The app uses the saved pipeline created by `python train_models.py`.")
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("No trained model found yet. Run `python train_models.py` from the project root, then refresh this page.")
else:
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    fields = meta["features"]
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(3)
        for i, field in enumerate(fields):
            col = cols[i % 3]
            if field in df.columns and (pd.api.types.is_numeric_dtype(df[field]) or pd.to_numeric(df[field], errors="coerce").notna().mean() > .9):
                vals = pd.to_numeric(df[field], errors="coerce").dropna()
                default = float(vals.median()) if len(vals) else 0.0
                values[field] = col.number_input(field, value=default)
            else:
                opts = sorted(df[field].dropna().astype(str).unique().tolist()) if field in df.columns else []
                values[field] = col.selectbox(field, opts) if opts else col.text_input(field)
        submitted = st.form_submit_button("Predict attrition risk")
    if submitted:
        sample = pd.DataFrame([values], columns=fields)
        proba = float(model.predict_proba(sample)[0,1])
        category = "High" if proba >= .70 else "Medium" if proba >= .40 else "Low"
        st.metric("Estimated attrition probability", f"{proba:.1%}")
        st.metric("Risk category (illustrative thresholds)", category)
        st.caption("Thresholds are illustrative and must be calibrated and validated before operational use. A risk score is not a judgment about an employee.")

st.divider()
st.subheader("Model performance")
metrics_path = ROOT / "outputs" / "model_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption("Metrics are generated from the held-out test set by the training script.")
else:
    st.info("Run training first to generate test-set metrics.")
