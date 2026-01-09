# src/baseline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ----------------------------
# Utilities
# ----------------------------
def _safe_rmse(y_true, y_pred) -> float:
    """
    RMSE computed without using mean_squared_error(squared=...),
    so it works across sklearn versions.
    """
    mse = mean_squared_error(y_true, y_pred)  # no squared kwarg
    return float(np.sqrt(mse))


def _infer_problem_type(y: pd.Series) -> str:
    """
    Simple fallback inference:
    - If y is numeric and has many unique values -> regression
    - Otherwise -> classification
    """
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "classification"

    if pd.api.types.is_numeric_dtype(y_nonnull):
        # Heuristic: many unique values -> regression
        nunique = int(y_nonnull.nunique(dropna=True))
        if nunique > max(20, int(0.05 * len(y_nonnull))):
            return "regression"
        # Could be 0/1 numeric labels -> classification
        return "classification"
    return "classification"


def _split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" prevents crashes on unseen categories in test
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ----------------------------
# Main baseline function
# ----------------------------
def build_and_eval_baseline(
    df: pd.DataFrame,
    target_col: str,
    problem_type: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Builds a simple baseline pipeline:
    - Preprocess: impute missing + scale numeric + one-hot categorical
    - Model:
        Classification: LogisticRegression (default), fallback to RandomForest if needed
        Regression: Ridge (default), fallback to RandomForestRegressor if needed
    Returns:
      {
        "metrics": {...},
        "model_name": "...",
        "notes": [...],
      }
    """
    notes: List[str] = []

    if not (0.05 <= float(test_size) <= 0.5):
        raise ValueError("test_size should be between 0.05 and 0.5 for a stable baseline.")

    X, y = _split_features_target(df, target_col)

    # Drop rows where target is missing (baseline should not try to learn from NaN target)
    before = len(y)
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    dropped = before - len(y)
    if dropped > 0:
        notes.append(f"Dropped {dropped} rows with missing target values.")

    if problem_type is None or str(problem_type).strip() == "":
        problem_type = _infer_problem_type(y)
        notes.append(f"Problem type not provided; inferred as '{problem_type}'.")
    else:
        problem_type = str(problem_type).strip().lower()
        if problem_type not in {"classification", "regression"}:
            notes.append(f"Unknown problem_type='{problem_type}', falling back to inference.")
            problem_type = _infer_problem_type(y)

    numeric_cols, categorical_cols = _get_column_types(X)
    if not numeric_cols:
        notes.append("No numeric columns detected.")
    if not categorical_cols:
        notes.append("No categorical columns detected.")
    if X.shape[1] == 0:
        raise ValueError("No feature columns available after removing the target column.")

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    # Train/test split
    stratify = None
    if problem_type == "classification":
        # Stratify only if there are at least 2 classes and enough samples
        y_unique = y.nunique(dropna=True)
        if y_unique >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=stratify
    )

    # Choose model
    if problem_type == "classification":
        # Baseline model: Logistic Regression
        model = LogisticRegression(max_iter=2000, n_jobs=None)
        model_name = "LogisticRegression"
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics: Dict[str, float] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            }

            # Optional AUC if binary and we have predict_proba
            try:
                classes = pd.Series(y_train).dropna().unique()
                if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(X_test)[:, 1]
                    # Need numeric/binary labels or consistent ordering; roc_auc_score handles label encoding internally
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                notes.append("Skipped ROC-AUC (not applicable or probability not available).")

            notes.append("Baseline is meant for a quick check, not final performance.")
            return {"metrics": metrics, "model_name": model_name, "notes": notes}

        except Exception as e:
            notes.append(f"LogisticRegression failed ({type(e).__name__}). Falling back to RandomForestClassifier.")

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=int(random_state),
                n_jobs=-1,
            )
            model_name = "RandomForestClassifier"
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            }
            notes.append("Random Forest is a strong general-purpose baseline.")
            return {"metrics": metrics, "model_name": model_name, "notes": notes}

    else:
        # Regression
        model = Ridge(random_state=int(random_state)) if "random_state" in Ridge().get_params() else Ridge()
        model_name = "Ridge"
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "rmse": _safe_rmse(y_test, y_pred),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
            notes.append("Baseline is meant for a quick check, not final performance.")
            return {"metrics": metrics, "model_name": model_name, "notes": notes}

        except Exception as e:
            notes.append(f"Ridge failed ({type(e).__name__}). Falling back to RandomForestRegressor.")

            model = RandomForestRegressor(
                n_estimators=300,
                random_state=int(random_state),
                n_jobs=-1,
            )
            model_name = "RandomForestRegressor"
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "rmse": _safe_rmse(y_test, y_pred),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
            notes.append("Random Forest is a strong general-purpose baseline for tabular regression.")
            return {"metrics": metrics, "model_name": model_name, "notes": notes}
