# REST API Documentation

## Start local service
From the project root:
```bash
python train_models.py
uvicorn api.main:app --reload
```

Base URL: `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints
- `GET /health` — returns API and model artifact availability.
- `GET /metadata` — returns target, feature list, and model name.
- `POST /predict` — returns a predicted attrition label, probability, and illustrative risk category.

## Example request
Use the exact features returned by `GET /metadata`; values below are illustrative only:
```json
{
  "features": {
    "Age": 34,
    "Department": "Sales"
  }
}
```
The example is intentionally incomplete if the model metadata requires more columns. The endpoint returns HTTP 422 and lists missing features. Supply every feature expected by the model.

## Example successful response
```json
{
  "predicted_attrition": "No",
  "attrition_probability": 0.1834,
  "illustrative_risk_category": "Low",
  "notice": "Educational decision-support output only; not a causal explanation or automated employment decision."
}
```
Numbers are examples, not real model output.

## HTTP status codes
- `200` — success
- `422` — request does not include all required features or fails validation
- `503` — model artifacts are not available yet

## Security notes
The starter API has no authentication. Do not expose it publicly until you add authentication, HTTPS, rate limiting, request logging with privacy controls, and a secrets-management strategy. Avoid logging employee names or other direct identifiers.
