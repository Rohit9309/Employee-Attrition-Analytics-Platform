
from pathlib import Path
from typing import Any
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

app = FastAPI(title="Employee Attrition Prediction API", version="1.0.0",
              description="Educational model endpoint. Not for automated employment decisions.")

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature names and values matching model metadata")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists() and META_PATH.exists()}

@app.get("/metadata")
def metadata():
    if not META_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metadata not found. Run python train_models.py first.")
    return json.loads(META_PATH.read_text(encoding="utf-8"))

@app.post("/predict")
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python train_models.py first.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    expected = meta["features"]
    missing = [c for c in expected if c not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    row = pd.DataFrame([{c: request.features[c] for c in expected}], columns=expected)
    probability = float(model.predict_proba(row)[0, 1])
    predicted = "Yes" if int(model.predict(row)[0]) == 1 else "No"
    risk = "High" if probability >= .70 else "Medium" if probability >= .40 else "Low"
    return {"predicted_attrition": predicted, "attrition_probability": round(probability, 4),
            "illustrative_risk_category": risk,
            "notice": "Educational decision-support output only; not a causal explanation or automated employment decision."}
