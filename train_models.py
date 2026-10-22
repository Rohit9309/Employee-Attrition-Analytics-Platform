
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "HR_Attrition_Indian_Dataset.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA).drop_duplicates()
if "LeftCompany" not in df.columns:
    raise ValueError("Expected target column 'LeftCompany'.")
df = df[df["LeftCompany"].isin(["Yes", "No"])].copy()
if "DateOfJoining" in df.columns:
    df["JoiningYear"] = pd.to_datetime(df["DateOfJoining"], errors="coerce").dt.year
    df.drop(columns=["DateOfJoining"], inplace=True)

drop_cols = [c for c in ["EmployeeID", "FullName", "LeftCompany"] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["LeftCompany"].map({"No": 0, "Yes": 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05, subsample=.9, colsample_bytree=.9, eval_metric="logloss", random_state=42, n_jobs=-1)
except Exception as e:
    print("XGBoost unavailable; continuing without it:", e)

results, fitted = [], {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, estimator in models.items():
    # GaussianNB requires dense one-hot features; convert output for this estimator.
    prep = preprocessor
    if name == "Gaussian Naive Bayes":
        # OneHotEncoder is sparse by default; wrap the final matrix in a dense conversion.
        from sklearn.preprocessing import FunctionTransformer
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
        ])
    pipe = Pipeline([("preprocess", prep), ("model", estimator)])
    try:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1": f1_score(y_test, pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, prob)})
        fitted[name] = pipe
        print(f"Trained {name}")
    except Exception as e:
        print(f"Skipped {name}: {e}")

metrics = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
metrics.to_csv(OUT / "model_metrics.csv", index=False)
if not fitted:
    raise RuntimeError("No models trained successfully.")
best_name = metrics.iloc[0]["Model"]
best = fitted[best_name]
joblib.dump(best, MODELS / "best_model.joblib")
metadata = {"best_model": best_name, "target": "LeftCompany", "positive_label": "Yes", "features": X.columns.tolist(), "categorical_features": categorical, "numeric_features": numeric}
(MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Cross-validation (use only compatible models and keep training data separate from test data).
cv_rows = []
for name, estimator in models.items():
    try:
        prep = preprocessor
        if name == "Gaussian Naive Bayes":
            prep = ColumnTransformer([
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
            ])
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}, n_jobs=1)
        row = {"Model": name}
        for metric in ["accuracy","precision","recall","f1","roc_auc"]:
            row[metric+"_mean"] = scores["test_"+metric].mean()
            row[metric+"_std"] = scores["test_"+metric].std()
        cv_rows.append(row)
    except Exception as e:
        print("CV skipped for", name, e)
pd.DataFrame(cv_rows).to_csv(OUT / "cross_validation.csv", index=False)

fig, axes = plt.subplots(1, len(fitted), figsize=(4*len(fitted), 4))
if len(fitted) == 1: axes = [axes]
for ax, (name, pipe) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(name, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=160); plt.close()

plt.figure(figsize=(7, 6))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name)
plt.plot([0,1], [0,1], "--", label="Chance")
plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=160); plt.close()

perm = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1)
importance = pd.DataFrame({"Feature": X_test.columns, "ImportanceMean": perm.importances_mean, "ImportanceStd": perm.importances_std}).sort_values("ImportanceMean", ascending=False)
importance.to_csv(OUT / "feature_importance.csv", index=False)

# Generate initial markdown evaluation report using actual scores from this run.
report = ["# Model Evaluation Report", "", "All scores below are calculated by running `python train_models.py`.", "", "## Held-out test-set comparison", "", metrics.round(4).to_markdown(index=False), "", f"## Selected model", "", f"Best model by held-out ROC-AUC: **{best_name}**.", "", "## Top permutation-importance features", "", importance.head(10).round(4).to_markdown(index=False), "", "## Interpretation notes", "", "- ROC-AUC measures ranking ability and does not guarantee well-calibrated probabilities.", "- Precision and recall should be considered together with the costs of false alerts and missed attrition.", "- Feature importance is predictive association, not causation.", "- This dataset may not represent a real organization's workforce."]
(OUT / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
print("\nTest metrics:\n", metrics.round(4).to_string(index=False))
print("Best model:", best_name)
print("Saved models, metrics, plots and report.")
