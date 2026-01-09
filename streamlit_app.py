import streamlit as st
import pandas as pd

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(page_title="ML Model Selector Advisor", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      /* Smaller header/title */
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
      .badge {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        margin-right: 0.35rem;
        border-radius: 999px;
        font-size: 0.82rem;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
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
# Header / Hero (smaller)
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <h1 style="margin: 0;">🧠 ML Model Selector Advisor</h1>
      <p class="muted" style="margin: 0.25rem 0 0.75rem 0;">
        Upload a CSV, pick a target, and get model recommendations + an optional baseline evaluation.
      </p>
      <span class="badge">Dataset profiling</span>
      <span class="badge">Model shortlist</span>
      <span class="badge">Baseline eval</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------
# Upload row
# ----------------------------
left, right = st.columns([1.2, 1])
with left:
    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])

with right:
    st.info("Tip: Pick a clean target column (avoid mixed types).", icon="💡")

if uploaded is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Target & Problem Type", expanded=True):
        target_col = st.selectbox("Target column", options=["(none)"] + list(df.columns))
        problem_hint = st.selectbox("Problem hint", ["Auto-detect", "Classification", "Regression"])

    with st.expander("Baseline Evaluation", expanded=True):
        run_baseline = st.toggle("Run baseline evaluation", value=True)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    st.divider()
    st.caption("Profiling → Recommendations → Baseline")

if target_col == "(none)":
    st.error("Select a target column in the sidebar to continue.")
    st.stop()

# ----------------------------
# Quick summary strip
# ----------------------------
non_null = int(df.notna().sum().sum())
total = int(df.shape[0] * df.shape[1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Filled cells", f"{non_null:,}")
m4.metric("Missing cells", f"{(total - non_null):,}")

st.write("")

# ----------------------------
# Compute profile + recommendations
# ----------------------------
with st.spinner("Analyzing dataset..."):
    profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)
    rec = recommend_models(profile)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Profile", "✅ Recommendations", "🚀 Baseline"])

with tab1:
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
        st.write("**Target column:**", target_col)
        st.write("**Type:**", str(tgt.dtype))
        st.write("**Missing:**", int(tgt.isna().sum()))
        st.write("**Unique:**", int(tgt.nunique(dropna=True)))

        # Safe, no-duplicate-column display for top classes
        if profile.get("problem_type") == "classification":
            st.write("**Top classes:**")
            vc = tgt.astype("string").value_counts(dropna=True).head(10)
            vc = vc.rename("count").reset_index()
            vc.columns = ["class", "count"]
            st.dataframe(vc, use_container_width=True, hide_index=True)
        else:
            st.write("**Target summary:**")
            # describe can return mixed types if target is not numeric; handle gracefully
            try:
                st.dataframe(tgt.describe().to_frame().T, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(tgt.astype("string").describe().to_frame().T, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Primary shortlist")

    shortlist_df = pd.DataFrame(rec.get("shortlist", []))
    if shortlist_df.empty:
        st.warning("No models were recommended. Check the profile output.")
    else:
        st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("Why these models?")
    rationale = rec.get("rationale", [])
    if not rationale:
        st.info("No rationale returned.")
    else:
        for r in rationale:
            st.info(r, icon="✅")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline evaluation")

    if not run_baseline:
        st.info("Baseline eval is turned off in the sidebar.")
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

            top1, top2, top3 = st.columns(3)
            top1.metric("Problem type", profile.get("problem_type", "unknown"))
            top2.metric("Model", results.get("model_name", "unknown"))
            top3.metric("Test size", f"{float(test_size):.2f}")

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

st.caption("—")
st.caption("Want it even cleaner? I can add download buttons + simple charts for missingness / class imbalance.")
