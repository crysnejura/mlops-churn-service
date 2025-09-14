import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "artifacts/model.pkl"
RANDOM_STATE = 32
TEST_SIZE = 0.2


def load_raw_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    df = df.dropna(subset=["TotalCharges"])

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def train_and_eval(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pre = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)

    print("Обучающие семплы:", len(X_train), "| Тестовые семплы:", len(X_test))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Accuracy:", round(acc, 4))


    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в: {MODEL_PATH}")


def main():
    df = load_raw_dataframe(DATA_PATH)
    train_and_eval(df)


if __name__ == "__main__":
    main()
