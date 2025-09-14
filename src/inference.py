import os
import joblib
import pandas as pd


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")

_model = None


def load_model(path: str = MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def _expected_columns(model) -> list[str]:
    pre = model.named_steps["pre"]
    return list(pre.feature_names_in_) 


def predict_proba_one(payload: dict, path: str = MODEL_PATH) -> float:
    model = load_model(path)
    cols = _expected_columns(model)

    x = pd.DataFrame([{c: payload.get(c) for c in cols}], columns=cols)

    return float(model.predict_proba(x)[:, 1][0])


def predict_label_one(payload: dict, threshold: float = 0.5, path: str = MODEL_PATH) -> dict:
    p = predict_proba_one(payload, path)
    return {
        "churn_probability": p,
        "prediction": "will_churn" if p >= threshold else "stay",
        "threshold": threshold,
    }