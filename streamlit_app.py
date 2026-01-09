import streamlit as st
import pandas as pd

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def simplify_rationale(rationale_list):
    mapping = {
        "nonlinear": "Your data has complex patterns (not straight-line relationships).",
        "linear": "Your data follows simpler, predictable patterns.",
        "small dataset": "This model works well even with limited data.",
        "large dataset": "This model scales well with large datasets.",
        "missing": "This model handles missing values well.",
        "categorical": "Your dataset contains categories (labels or names), and this model works well with them.",
        "high dimensional": "You have many columns, and this model can handle that.",
        "imbalanced": "Your target labels are uneven, and this model handles that well.",
        "baseline": "This is a good starting model for beginners.",
        "overfit": "This choice helps reduce overfitting (memorizing instead of learning).",
        "noise": "This model tends to handle noisy real-world data well.",
        "outlier": "This model is less sensitive to extreme values (outliers).",
    }

    results = []
    for r in rationale_list or []:
        r_low = str(r).lower()
        for k, v in mapping.items():
            if k in r_low:
                results.append(v)
                break
        else:
            results.append("This model matches your dataset characteristics well.")
    return results


def problem_type_sentence(profile, df, target_col):
    pt = str(profile.get("problem_type", "")).lower()
    y = df[target_col]

    if pt == "classification":
        n = y.nunique(dropna=True)
        examples = ", ".join(y.dropna().astype(str).unique()[:4])
        return (
            f"This is a **classification problem** because your target column "
            f"**'{target_col}'** has **{n} distinct label(s)** "
            f"(for example: {examples})."
        )

    if pt == "regression":
        return (
            f"This is a **regression problem** because your target column "
            f"**'{target_col}'** contains numeric values that vary continuously."
        )

    return "The problem type could not be confidently determined."


def format_advanced_block(profile, rec):
    """
    Shows technical rationale + a few key dataset signals (if available).
    This is intentionally flexible because profile schema can vary.
    """
    lines = []

    # Technical rationale from recommender (as-is)
    tech = rec.get("rationale", []) or []
    if tech:
        lines.append("**Technical rationale (from the recommender):**")
        for t in tech:
            lines.append(f"- {t}")
    else:
        lines.append("**Technical rationale:** (none provided)")

    # Key dataset signals (best-effort, only if present)
    lines.append("")
    lines.append("**Key dataset signals used (if available):**")
    keys_of_interest = [
        "problem_type",
        "n_rows",
        "n_cols",
        "target_dtype",
        "target_nunique",
        "missingness_overall",
        "class_balance",
        "n_numeric",
        "n_categorical",
    ]

    shown_any = False
    for k in keys_of_interest:
        if k in profile:
            lines.append(f"- `{k}`: {profile[k]}")
            shown_any = True

    if not shown_any:
        # If schema is unknown, show a compact view of profile keys (but not everything)
        preview_keys = list(profile.keys())[:12]
        lines.append(f"- Available profile keys: {preview_keys} (schema depends on your profiling module)")

    return "\n".join(lines)


# -------------------------------------------------
# Page config + styles
# -------------------------------------------------
st.set_page_config(page_title="ML Model Selector Advisor", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.25rem; }
    h3 { font-size: 1.05rem; }
    .hero {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.15);
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.1));
    }
    .card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 ML Model Selector Advisor</h1>
        <p>Upload a CSV, select a target, and get beginner-friendly model recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -------------------------------------------------
# Upload
# -------------------------------------------------
left, right = st.columns([1.2, 1])
with left:
    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
with right:
    st.info("Choose the column you want to predict.", icon="💡")

if uploaded is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    target_col = st.selectbox("Target column", ["(none)"] + list(df.columns))
    problem_hint = st.selectbox("Problem hint", ["Auto-detect", "Classification", "Regression"])

    run_baseline = st.toggle("Run baseline evaluation", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42)

if target_col == "(none)":
    st.error("Select a target column to continue.")
    st.stop()

# -------------------------------------------------
# Dataset summary
# -------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Rows", df.shape[0])
m2.metric("Columns", df.shape[1])
m3.metric("Missing cells", int(df.isna().sum().sum()))

st.write("")

# -------------------------------------------------
# Compute
# -------------------------------------------------
with st.spinner("Analyzing dataset..."):
    profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)
    rec = recommend_models(profile)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["✅ Recommendations", "📊 Dataset Profile", "🚀 Baseline"])

# ---------------- TAB 1 ----------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recommended models")

    shortlist_df = pd.DataFrame(rec.get("shortlist", []))
    st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("Why these models were chosen (simple explanation)")

    st.info(problem_type_sentence(profile, df, target_col), icon="🎯")

    reasons = simplify_rationale(rec.get("rationale", []))
    for r in dict.fromkeys(reasons):
        st.info(r, icon="💡")

    # ✅ NEW: Advanced explanation (optional)
    with st.expander("Advanced explanation (optional)"):
        st.markdown(format_advanced_block(profile, rec))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset profile")
    st.json(profile)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline evaluation")

    if run_baseline:
        try:
            results = build_and_eval_baseline(
                df=df,
                target_col=target_col,
                problem_type=profile.get("problem_type"),
                test_size=float(test_size),
                random_state=int(random_state),
            )

            st.metric("Model used", results.get("model_name"))
            st.dataframe(pd.DataFrame([results.get("metrics", {})]), hide_index=True)

            for n in results.get("notes", []):
                st.write("-", n)

        except Exception as e:
            st.error(f"Baseline failed: {e}")
    else:
        st.info("Baseline evaluation is disabled.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("— ML Model Selector Advisor")
