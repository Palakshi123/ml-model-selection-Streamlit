def recommend_models(profile: dict) -> dict:
    n = profile["n_rows"]
    ptype = profile["problem_type"]
    miss = profile["missing_rate"]
    high_card = len(profile["high_cardinality_categoricals"]) > 0
    imbalance = profile.get("class_imbalance_ratio")

    rationale = []
    shortlist = []

    # General signals
    if miss and miss > 0.1:
        rationale.append(f"Missingness is {miss:.0%}; favor models/pipelines robust to missing values + imputation.")
    if high_card:
        rationale.append("High-cardinality categorical features detected; prefer target encoding/one-hot w/ regularization or tree ensembles after encoding.")
    if n < 2000:
        rationale.append("Small dataset; simpler models + regularization often generalize better.")
    elif n > 200000:
        rationale.append("Large dataset; consider scalable linear models and histogram-based GBDTs.")

    if ptype == "classification":
        if imbalance and imbalance >= 3:
            rationale.append(f"Imbalance ratio ~{imbalance:.1f}; use class_weight, calibrated thresholds, PR-AUC metrics.")
        # Shortlist
        shortlist.extend([
            {"model": "Logistic Regression (L2)", "when": "Strong baseline, interpretable, fast", "notes": "Use class_weight for imbalance; one-hot/standardize."},
            {"model": "Linear SVM (hinge)", "when": "High-dimensional sparse features (one-hot)", "notes": "Works well with text/bag-of-words style data."},
            {"model": "Random Forest", "when": "Nonlinearities + interactions, medium data", "notes": "Less sensitive to scaling; can overfit small n."},
            {"model": "Gradient Boosting (HistGradientBoosting)", "when": "Tabular data, strong performance", "notes": "Great default; tune depth/learning_rate."},
        ])
        if n > 100000:
            shortlist.append({"model": "SGDClassifier (log loss)", "when": "Very large data", "notes": "Fast incremental training; needs feature scaling."})

    else:  # regression
        shortlist.extend([
            {"model": "Ridge Regression", "when": "Strong baseline, multicollinearity", "notes": "Standardize numeric; one-hot categoricals."},
            {"model": "Elastic Net", "when": "Need sparsity + stability", "notes": "Useful when many features; tune l1_ratio."},
            {"model": "Random Forest Regressor", "when": "Nonlinear patterns, medium data", "notes": "Robust baseline but can be heavy."},
            {"model": "HistGradientBoostingRegressor", "when": "Tabular data, best default", "notes": "Strong accuracy; tune depth/learning_rate."},
        ])
        if n > 100000:
            shortlist.append({"model": "SGDRegressor", "when": "Very large data", "notes": "Scales well; needs careful scaling/tuning."})

    # Prioritize a “primary” shortlist (top 3-5)
    # Simple ordering rule: prefer GBDT + linear baseline always; add RF if medium.
    primary = []
    if ptype == "classification":
        primary = [shortlist[3], shortlist[0], shortlist[2]]  # HGB, LogReg, RF
    else:
        primary = [shortlist[3], shortlist[0], shortlist[2]]  # HGB, Ridge, RF

    return {
        "shortlist": primary,
        "rationale": rationale if rationale else ["Standard tabular setup detected; start with a strong baseline + a boosting model."],
    }
