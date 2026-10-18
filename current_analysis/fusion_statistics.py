from __future__ import annotations
import argparse

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

PROJECT_DIR = None
INPUT = None
OUTPUT_DIR = None
SEED = 20260718
N_BOOTSTRAP = 2000
MODELS = {
    "Clinical-only": "Clinical-only_Score",
    "Clinical_RadDiff_DLHet": "Clinical_RadDiff_DLHet_Score",
    "Four-score_without_Habitat": "Four-score_without_Habitat_Score",
    "Full_five-score_Fusion": "Full_five-score_Fusion_Score",
    "Habitat_max-H2": "Habitat_Score",
}
COMPARISON = ("Full_five-score_Fusion", "Four-score_without_Habitat")


def cindex(frame: pd.DataFrame, score: str) -> float:
    return float(concordance_index(frame["duration"], -frame[score], frame["event"]))


def bootstrap(frame: pd.DataFrame, score_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = len(frame)
    for replicate in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        sample = frame.iloc[indices]
        if sample["event"].nunique() < 2 or sample["duration"].nunique() < 2:
            continue
        values = {column: cindex(sample, column) for column in score_columns}
        rows.append({"replicate": replicate, **values, "delta_full_minus_four": values[MODELS[COMPARISON[0]]] - values[MODELS[COMPARISON[1]]]})
    return pd.DataFrame(rows)


def quantile_interval(values: pd.Series) -> tuple[float, float]:
    return float(values.quantile(0.025)), float(values.quantile(0.975))


def risk_stratification(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame.loc[frame["group"] == "train"]
    rows = []
    for model, score in MODELS.items():
        cutoff = float(train[score].median())
        for group in ("train", "test1", "test2"):
            subset = frame.loc[frame["group"] == group].copy()
            high = subset[score] >= cutoff
            result = logrank_test(subset.loc[high, "duration"], subset.loc[~high, "duration"], subset.loc[high, "event"], subset.loc[~high, "event"])
            fit = subset[["duration", "event", score]].copy()
            fit["high_risk"] = high.astype(int)
            cox = CoxPHFitter(penalizer=0.01).fit(fit[["duration", "event", "high_risk"]], "duration", "event")
            rows.append({"model": model, "group": group, "endpoint": "PFS exploratory migration" if group == "test2" else "RFS", "train_median_cutoff": cutoff, "n_high": int(high.sum()), "n_low": int((~high).sum()), "events_high": int(subset.loc[high, "event"].sum()), "events_low": int(subset.loc[~high, "event"].sum()), "hr_high_vs_low": float(np.exp(cox.params_["high_risk"])), "hr_ci_lower": float(np.exp(cox.confidence_intervals_.loc["high_risk"].iloc[0])), "hr_ci_upper": float(np.exp(cox.confidence_intervals_.loc["high_risk"].iloc[1])), "logrank_p": float(result.p_value)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    summary_rows = []
    for group in ("train", "test1", "test2"):
        subset = frame.loc[frame["group"] == group].reset_index(drop=True)
        replicates = bootstrap(subset, list(MODELS.values()), rng)
        replicates.insert(0, "group", group)
        bootstrap_rows.append(replicates)
        for model, score in MODELS.items():
            lower, upper = quantile_interval(replicates[score])
            summary_rows.append({"model": model, "group": group, "endpoint": "PFS exploratory migration" if group == "test2" else "RFS", "n": len(subset), "events": int(subset["event"].sum()), "c_index": cindex(subset, score), "ci_lower": lower, "ci_upper": upper, "successful_bootstrap": len(replicates)})
        delta_lower, delta_upper = quantile_interval(replicates["delta_full_minus_four"])
        summary_rows.append({"model": "Delta_Full_minus_Four", "group": group, "endpoint": "PFS exploratory migration" if group == "test2" else "RFS", "n": len(subset), "events": int(subset["event"].sum()), "c_index": cindex(subset, MODELS[COMPARISON[0]]) - cindex(subset, MODELS[COMPARISON[1]]), "ci_lower": delta_lower, "ci_upper": delta_upper, "successful_bootstrap": len(replicates)})
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(OUTPUT_DIR / "bootstrap_replicates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "cindex_bootstrap_ci_and_delta.csv", index=False, encoding="utf-8-sig")
    risk_stratification(frame).to_csv(OUTPUT_DIR / "fixed_train_cutoff_risk_stratification.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "final_statistics_manifest.json").write_text(json.dumps({"status": "complete", "seed": SEED, "bootstrap_replicates_requested": N_BOOTSTRAP, "comparison": "Full five-score minus four-score without Habitat", "cutoff_policy": "Median score in Train, frozen and applied unchanged to Test1/Test2", "test2_policy": "Exploratory PFS migration"}, ensure_ascii=False, indent=2), encoding="utf-8")



def configure_cli():
    global INPUT, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Fixed-score bootstrap discrimination and risk stratification")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    INPUT, OUTPUT_DIR = args.scores.resolve(), args.output.resolve()


if __name__ == "__main__":
    configure_cli()
    main()
