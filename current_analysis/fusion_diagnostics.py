"""PH diagnostics and clinical-plus-fusion validation, with a legacy fusion-only option."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index


def standardized_training(scores, model):
    train = scores.loc[scores["group"] == "train"].reset_index(drop=True)
    frame = pd.DataFrame({p + "_Z": (train[p] - model["scalers"][p]["mean"]) /
        model["scalers"][p]["scale"] for p in model["predictors"]})
    return pd.concat([frame, train[["duration", "event"]]], axis=1)


def fit_checked(frame, spec):
    fit = CoxPHFitter(penalizer=spec["penalizer"]).fit(frame, "duration", "event")
    error = max(abs(float(fit.params_[p]) - b) for p, b in spec["coefficients"].items())
    if error > 2e-6:
        raise ValueError(f"Training refit disagrees with locked coefficients: {error}")
    return fit


def ph_diagnostics(scores, models):
    rows = []
    for name, spec in models.items():
        frame = standardized_training(scores, spec)
        fit = fit_checked(frame, spec)
        result = proportional_hazard_test(fit, frame, time_transform=["rank", "km"]).summary.reset_index()
        for _, r in result.iterrows():
            rows.append({"model": name, "diagnostic_target": "locked multivariable score model",
                "term": r.iloc[0], "time_transform": r.iloc[1],
                "chi_square": float(r["test_statistic"]), "p": float(r["p"])})
        train = scores.loc[scores["group"] == "train", ["duration", "event", name + "_Score"]].copy()
        single = CoxPHFitter(penalizer=.01).fit(train, "duration", "event")
        result = proportional_hazard_test(single, train, time_transform=["rank", "km"]).summary.reset_index()
        for _, r in result.iterrows():
            rows.append({"model": name, "diagnostic_target": "training Cox model with locked score as sole predictor",
                "term": r.iloc[0], "time_transform": r.iloc[1],
                "chi_square": float(r["test_statistic"]), "p": float(r["p"])})
    return pd.DataFrame(rows)


def bootstrap_optimism(scores, spec, resamples=1000, seed=7):
    """Refit only the fusion stage; fixed component scores and Train scaling."""
    if resamples < 1:
        raise ValueError("The bootstrap count must be positive")
    frame = standardized_training(scores, spec)
    fit = fit_checked(frame, spec)
    apparent = float(concordance_index(frame.duration, -fit.predict_log_partial_hazard(frame), frame.event))
    rng = np.random.default_rng(seed)
    optimism, failures = [], []
    for i in range(resamples):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)
        try:
            model = CoxPHFitter(penalizer=spec["penalizer"]).fit(sample, "duration", "event")
            cb = concordance_index(sample.duration, -model.predict_log_partial_hazard(sample), sample.event)
            co = concordance_index(frame.duration, -model.predict_log_partial_hazard(frame), frame.event)
            optimism.append(float(cb - co))
        except Exception as exc:
            failures.append({"replicate": i + 1, "error": type(exc).__name__})
    if len(optimism) < .99 * resamples:
        raise ValueError("More than 1% of bootstrap replicates failed")
    mean = float(np.mean(optimism))
    return {"apparent": apparent, "mean_optimism": mean, "corrected": apparent - mean,
        "requested_replicates": resamples, "successful_replicates": len(optimism), "seed": seed,
        "penalizer": spec["penalizer"], "component_scores_fixed": True,
        "entire_feature_selection_pipeline_repeated": False, "failures": failures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--clinical-input", type=Path)
    args = p.parse_args()
    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    models = metadata.get("model_metadata", metadata.get("fusion_models"))
    if not models:
        raise ValueError("No locked model metadata")
    scores = pd.read_csv(args.scores)
    args.output.mkdir(parents=True, exist_ok=True)
    ph_diagnostics(scores, models).to_csv(args.output / "ph_numeric.csv", index=False)
    if args.clinical_input is None:
        result = bootstrap_optimism(scores, models["Full_five-score_Fusion"], args.bootstrap)
        result["scope_warning"] = "Fusion-only diagnostic; not the corrected manuscript bootstrap"
    else:
        from .clinical_fusion_validation import bootstrap_clinical_fusion
        from .fusion_locked_core import CLINICAL_VARS
        raw = pd.read_csv(args.clinical_input)
        joined = scores.merge(raw[["ID"] + CLINICAL_VARS], on="ID", validate="one_to_one")
        records = bootstrap_clinical_fusion(joined[joined.group == "train"], args.bootstrap)
        successful = [r for r in records if not r.get("error")]
        training = joined[joined.group == "train"]
        apparent = float(concordance_index(training.duration, -training["Full_five-score_Fusion_Score"], training.event))
        mean = float(np.mean([r["optimism"] for r in successful]))
        result = {"apparent": apparent, "mean_optimism": mean, "corrected": apparent - mean,
                  "requested_replicates": args.bootstrap, "successful_replicates": len(successful),
                  "warnings_in_resamples": sum(bool(r.get("warnings")) for r in records),
                  "scope": "Refit clinical and fusion stages; four image scores fixed", "records": records}
    (args.output / "bootstrap_optimism.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
