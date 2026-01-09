import streamlit as st
import pandas as pd

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

# ----------------------------
# Helpers (beginner-friendly explanations)
# ----------------------------
def simplify_rationale(rationale_list):
    mapping = {
        "nonlinear": "Your data has complex patterns (not straight-line relationships).",
        "linear": "Your data likely follows simpler, more predictable patterns.",
        "small dataset": "This model can work well even if you don’t have a lot of data.",
        "large dataset": "This model can scale well when you have lots of rows.",
        "missing": "This model can handle missing values better than many others.",
        "categorical": "Your dataset has categories (like names/labels), and this model can work well with them.",
        "high dimensional": "You have many columns/features, and this model can handle that well.",
        "sparse": "Your data has lots of zeros/empty values, and this model can work well in that case.",
        "imbalanced": "Your target classes are uneven, and this model is often a good choice for that.",
        "interpret": "This model is easier to understand and explain to others.",
        "baseline": "This is a great starting point before trying more advanced models.",
        "overfit": "This model choice helps reduce overfitting (memorizing instead of learning).",
        "noise": "This model tends to handle noisy real-world data well.",
        "outlier": "This model is less sensitive to outliers (extreme values).",
    }

    simplified = []
    for r in (rationale_list or []):
        lower = str(r).lower()
        matched = False
        for key, explanation in mapping.items():
            if key in lower:
                simplified.append(explanation)
                matched = True
                break
        if not matched:
            simplified.append("This model matches your dataset well and is a strong practical starting point.")
    return simplified


MODEL_EXPLANATIONS = {
    "Logistic Regression": "Simple and fast. Great first model for classification.",
    "Linear Regression": "Great when the target changes steadily with the inputs.",
    "Ridge Regression": "Like linear regression, but more stable when features are correlated.",
    "Lasso Regression": "Like linear regression, but can automatically drop less useful features.",
    "Decision Tree": "Easy to understand. Can overfit if data is noisy.",
    "Random Forest": "Strong all-rounder. Works well on many real datasets with little tuning.",
    "Gradient Boosting": "Often more accurate than Random Forest, but can be slower.",
    "XGBoost": "Very powerful on tabular data; great when accuracy matters.",
    "LightGBM": "Fast boosting model, great for larger tabular datasets.",
    "CatBoost": "Boosting model that works especially well with categorical features.",
    "SVM": "Good for smaller datasets with clean separation; can be slow on large data.",
    "KNN": "Simple idea (similar neighbors). Can be slow with large datasets.",
    "Naive Bayes": "Very fast for text-like or count-like features; good baseline classifier.",
}

def problem_type_sentence(profile, df, target_col) -> str:
    """
    Short beginner-friendly sentence that explains classification vs regression
    using the target column characteristics.
    """
    pt = str(profile.get("problem_type", "unknown")).lower()
    y = df[target_col]

    if pt == "classification":
        n = int(y.nunique(dropna=True))
        examples = y.dropna().astype(str).unique()[:5]
        ex_txt = ", ".join(examples) if len(examples) else "N/A"
        return (
            f"This is a **classification** problem because your target **'{target_col}'** has "
            f"**{n} unique label(s)** (e.g., {ex_txt})."
        )

    if pt == "regression":
        y_nonnull = y.dropna()
        # Try to show a tiny numeric range if possible
        if pd.api.types.is_numeric_dtype(y_nonnull) and len(y_nonnull) > 0:
            try:
                y_min = float(y_nonnull.min())
                y_max = float(y_nonnull.max())
                return (
                    f"This is a **regression** problem because your target **'{target_col}'** is numeric "
                    f"and varies continuously (roughly from **{y_min:.3g}** to **{y_max:.3g}**)."
                )
            except Exception:
                pass

        return (
            f"This is a **regression** problem because your target **'{target_col}'** is numeric "
            "and varies continuously."
        )

    return "Problem type could not be determined confidently from the target column."

# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(page_title="ML Model Selector Advisor", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      h1 {font-size: 1.55rem !important; margin-bottom: 0.25rem;}
      h2 {font-size: 1.2rem !important;}
      h3 {font-size: 1.05rem !important;}
      .muted {opacity: 0.8;}

      .hero {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(16,185,129,0.10));
      }
      .card {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
        padding: 1rem;
      }
      .stAlert > div {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header (no badges)
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">🧠 ML Model Selector Advisor</h1>
      <p class="muted" style="margin:0.25rem 0 0 0;">
