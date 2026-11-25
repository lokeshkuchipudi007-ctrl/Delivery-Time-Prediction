"""Train the delivery time model on synthetic data and save it to model.pkl.

Replace `generate_data` with your real dataset loader when you have one.
The feature names must stay in sync with FEATURES in app.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = Path(__file__).parent / "model.pkl"

NUMERIC = ["distance_km", "prep_time_min", "courier_experience_years"]
CATEGORICAL = ["traffic", "weather"]
TARGET = "delivery_time_min"


def generate_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "distance_km": rng.uniform(0.5, 20, n).round(2),
            "prep_time_min": rng.integers(5, 40, n),
            "courier_experience_years": rng.uniform(0, 10, n).round(1),
            "traffic": rng.choice(["low", "medium", "high"], n, p=[0.3, 0.45, 0.25]),
            "weather": rng.choice(["clear", "rain", "storm"], n, p=[0.7, 0.25, 0.05]),
        }
    )
    traffic_penalty = df["traffic"].map({"low": 0, "medium": 5, "high": 12})
    weather_penalty = df["weather"].map({"clear": 0, "rain": 4, "storm": 10})
    df[TARGET] = (
        df["prep_time_min"]
        + df["distance_km"] * 2.5
        + traffic_penalty
        + weather_penalty
        - df["courier_experience_years"] * 0.6
        + rng.normal(0, 2.5, n)
    ).clip(lower=5).round(1)
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )


def main() -> None:
    df = generate_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = build_pipeline().fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f} min")
    print(f"R2 : {r2_score(y_test, preds):.3f}")
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
