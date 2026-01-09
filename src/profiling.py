import pandas as pd
import numpy as np

def _infer_problem_type(y: pd.Series) -> str:
    # Heuristic:
    # - numeric with many unique values -> regression
    # - otherwise -> classification
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique > 20:
            return "regression"
    return "classification"

def profile_dataset(df: pd.DataFrame, target_col: str, problem_hint: str = "Auto-detect") -> dict:
    if target_col not in df.columns:
        raise ValueError("target_col not found in dataframe")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    n_rows, n_cols = df.shape
    missing_rate = float(df.isna().mean().mean())

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # cardinality
    high_card_cols = []
    for c in cat_cols:
        if X[c].nunique(dropna=True) > min(50, max(10, int(0.2 * n_rows))):
            high_card_cols.append(c)

    # target analysis
    if problem_hint == "Classification":
        problem_type = "classification"
    elif problem_hint == "Regression":
        problem_type = "regression"
    else:
        problem_type = _infer_problem_type(y)

    class_imbalance = None
    n_classes = None
    if problem_type == "classification":
        vc = y.value_counts(dropna=True)
        n_classes = int(vc.shape[0])
        if len(vc) > 1:
            class_imbalance = float(vc.max() / vc.min())
        else:
            class_imbalance = 1.0

    # outliers proxy (numeric)
    outlier_proxy = None
    if numeric_cols:
        sample = X[numeric_cols].select_dtypes(include=[np.number]).copy()
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        if not sample.empty:
            # robust z proxy using IQR
            q1 = sample.quantile(0.25)
            q3 = sample.quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)
            outlier_proxy = float(((sample < (q1 - 1.5 * iqr)) | (sample > (q3 + 1.5 * iqr))).mean().mean())

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_features": int(X.shape[1]),
        "missing_rate": missing_rate,
        "n_numeric": int(len(numeric_cols)),
        "n_categorical": int(len(cat_cols)),
        "high_cardinality_categoricals": high_card_cols,
        "problem_type": problem_type,
        "n_classes": n_classes,
        "class_imbalance_ratio": class_imbalance,  # ~1 is balanced; >3 starts getting spicy
        "outlier_proxy_rate": outlier_proxy,
    }
