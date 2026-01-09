import streamlit as st
import pandas as pd

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

# ----------------------------
# Helpers (beginner-friendly explanations)
# ----------------------------
def simplify_rationale(rationale_list):
    """
    Converts technical rationale strings into beginner-friendly explanations.
    If no match is found, returns a safe generic explanation.
    """
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
# Header (NO badges)
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">🧠 ML Model Selector Advisor</h1>
      <p class="muted" style="margin:0.25rem 0 0 0;">
        Upload a CSV, select a target, and get beginner-friendly model suggestions.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------
# Upload
# ----------------------------
left, right = st.columns([1.2, 1])
with left:
    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
with right:
    st.info("Tip: Choose a target column you want to predict.", icon="💡")

if uploaded is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Target & Problem Type", expanded=True):
        target_col = st.selectbox("Target column", ["(none)"] + list(df.columns))
        problem_hint = st.selectbox("Problem hint", ["Auto-detect", "Classification", "Regression"])

    with st.expander("Baseline Evaluation", expanded=True):
        run_baseline = st.toggle("Run baseline evaluation", value=True)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

if target_col == "(none)":
    st.error("Select a target column to continue.")
    st.stop()

# ----------------------------
# Dataset summary strip
# ----------------------------
non_null = int(df.notna().sum().sum())
total = int(df.shape[0] * df.shape[1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Filled", f"{non_null:,}")
m4.metric("Missing", f"{(total - non_null):,}")

st.write("")

# ----------------------------
# Compute profile + recommendations
# ----------------------------
with st.spinner("Analyzing dataset..."):
    profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)
    rec = recommend_models(profile)

# ----------------------------
# Tabs (Tab 1 = Recommendations)
# ----------------------------
tab1, tab2, tab3 = st.tabs(["✅ Recommendations", "📊 Dataset Profile", "🚀 Baseline Evaluation"])

# ===== TAB 1: RECOMMENDATIONS =====
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recommended models")

    shortlist_df = pd.DataFrame(rec.get("shortlist", []))
    if shortlist_df.empty:
        st.warning("No models recommended. Try checking your target column or problem hint.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

        st.write("")
        st.subheader("Why these models were chosen (simple explanation)")
        simple_reasons = simplify_rationale(rec.get("rationale", []))
        # Avoid repeating identical messages
        seen = set()
        for reason in simple_reasons:
            if reason not in seen:
                st.info(reason, icon="💡")
                seen.add(reason)

        st.write("")
        st.subheader("Quick model guide (for beginners)")
        # Try to find a sensible column that contains model names
        model_col = None
        for c in shortlist_df.columns:
            if c.lower() in {"model", "model_name", "name", "algorithm"}:
                model_col = c
                break

        if model_col is None:
            st.caption("Tip: If your shortlist includes a model column, I'll show model-by-model explanations here.")
        else:
            for m in shortlist_df[model_col].astype(str).head(8).tolist():
                expl = MODEL_EXPLANATIONS.get(m, "A strong general-purpose choice for datasets like yours.")
                st.write(f"• **{m}** — {expl}")

        st.markdown("</div>", unsafe_allow_html=True)

# ===== TAB 2: PROFILE =====
with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset profile")
        st.json(profile)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Target preview")

        tgt = df[target_col]
        st.write("Target:", target_col)
        st.write("Type:", str(tgt.dtype))
        st.write("Missing:", int(tgt.isna().sum()))
        st.write("Unique:", int(tgt.nunique(dropna=True)))

        # Safe class counts (no duplicate columns; arrow-safe)
        if profile.get("problem_type") == "classification":
            st.write("Top classes:")
            vc = tgt.astype("string").value_counts(dropna=True).head(10)
            vc = vc.rename("count").reset_index()
            vc.columns = ["class", "count"]
            st.dataframe(vc, use_container_width=True, hide_index=True)
        else:
            st.write("Target summary:")
            try:
                st.dataframe(tgt.describe().to_frame().T, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(
                    tgt.astype("string").describe().to_frame().T,
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

# ===== TAB 3: BASELINE =====
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline evaluation")

    if not run_baseline:
        st.info("Baseline evaluation is disabled in the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Quick baseline metrics for a fast sanity check (not full tuning).")

        try:
            with st.spinner("Running baseline..."):
                results = build_and_eval_baseline(
                    df=df,
                    target_col=target_col,
                    problem_type=profile.get("problem_type"),
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Problem type", profile.get("problem_type", "unknown"))
            c2.metric("Model", results.get("model_name", "unknown"))
            c3.metric("Test size", f"{float(test_size):.2f}")

            st.write("### Metrics")
            metrics = results.get("metrics", {})
            if metrics:
                st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)
            else:
                st.warning("No metrics returned from baseline.")

            st.write("### Notes")
            notes = results.get("notes", [])
            if notes:
                for n in notes:
                    st.write(f"- {n}")
            else:
                st.caption("No notes returned.")

        except Exception as e:
            st.error(f"Baseline failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

st.caption("— ML Model Selector Advisor")
