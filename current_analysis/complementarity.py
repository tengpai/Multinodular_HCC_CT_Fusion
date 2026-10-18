from __future__ import annotations
import argparse

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_DIR = None
INPUT_PATH = None
FUSION_MANIFEST_PATH = None
OUTPUT_DIR = None
SEED = 20260718
N_PERMUTATIONS = 1000
PENALIZER = 0.01
SCORES = [
    "Clinical_Score",
    "Rad_Diff_HighRisk",
    "DL_Heterogeneity",
    "DL_MaxPooling",
    "Habitat_Score",
]
DISPLAY_NAMES = {
    "Clinical_Score": "Clinical",
    "Rad_Diff_HighRisk": "Rad-Diff",
    "DL_Heterogeneity": "DL-Heterogeneity",
    "DL_MaxPooling": "DL-MaxPooling",
    "Habitat_Score": "Habitat-max-H2",
}
GROUPS = ("train", "test1", "test2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def c_index(frame: pd.DataFrame, risk: np.ndarray) -> float:
    return float(concordance_index(frame["duration"], -np.asarray(risk, dtype=float), frame["event"]))


def standardize_from_train(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    train = frame.loc[frame["group"] == "train"]
    standardized = frame.copy()
    metadata = {}
    for score in SCORES:
        mean = float(train[score].mean())
        scale = float(train[score].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid Train scale for {score}")
        standardized[f"{score}_Z"] = (frame[score] - mean) / scale
        metadata[score] = {"mean": mean, "scale": scale}
    return standardized, metadata


def fit_joint_cox(frame: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    z_columns = [f"{score}_Z" for score in SCORES]
    train = frame.loc[frame["group"] == "train", ["duration", "event", *z_columns]].copy()
    model = CoxPHFitter(penalizer=PENALIZER)
    model.fit(train, duration_col="duration", event_col="event")
    summary = model.summary.reset_index().rename(columns={"covariate": "score_z"})
    summary["score"] = summary["score_z"].str.removesuffix("_Z")
    summary["display_name"] = summary["score"].map(DISPLAY_NAMES)
    summary["independent_at_p_lt_0_05"] = summary["p"] < 0.05
    columns = [
        "score", "display_name", "coef", "exp(coef)", "se(coef)",
        "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p",
        "independent_at_p_lt_0_05",
    ]
    return model, summary[columns]


def correlation_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["group"] == "train", SCORES]
    matrix = train.corr(method="pearson")
    pairs = []
    for left_index, left in enumerate(SCORES):
        for right in SCORES[left_index + 1:]:
            pairs.append({
                "score_1": DISPLAY_NAMES[left],
                "score_2": DISPLAY_NAMES[right],
                "pearson_r": float(matrix.loc[left, right]),
                "absolute_r": abs(float(matrix.loc[left, right])),
            })
    return matrix.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES), pd.DataFrame(pairs)


def vif_output(frame: pd.DataFrame) -> pd.DataFrame:
    z_columns = [f"{score}_Z" for score in SCORES]
    train_values = frame.loc[frame["group"] == "train", z_columns].to_numpy(dtype=float)
    return pd.DataFrame([
        {
            "score": score,
            "display_name": DISPLAY_NAMES[score],
            "vif": float(variance_inflation_factor(train_values, index)),
        }
        for index, score in enumerate(SCORES)
    ])


def analytic_shap(frame: pd.DataFrame, coefficients: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows = []
    summary_rows = []
    for group in GROUPS:
        subset = frame.loc[frame["group"] == group].copy()
        group_contributions = {}
        for score in SCORES:
            contribution = subset[f"{score}_Z"].to_numpy(dtype=float) * float(coefficients[f"{score}_Z"])
            group_contributions[score] = contribution
            for patient_id, value in zip(subset["ID"], contribution):
                patient_rows.append({
                    "ID": patient_id,
                    "group": group,
                    "score": score,
                    "display_name": DISPLAY_NAMES[score],
                    "shap_contribution": float(value),
                    "absolute_shap": abs(float(value)),
                })
        mean_absolute = {score: float(np.mean(np.abs(values))) for score, values in group_contributions.items()}
        total = sum(mean_absolute.values())
        ranked = sorted(SCORES, key=lambda score: (-mean_absolute[score], score))
        for rank, score in enumerate(ranked, start=1):
            summary_rows.append({
                "group": group,
                "score": score,
                "display_name": DISPLAY_NAMES[score],
                "mean_absolute_shap": mean_absolute[score],
                "relative_share": mean_absolute[score] / total if total > 0 else np.nan,
                "rank": rank,
            })
    return pd.DataFrame(patient_rows), pd.DataFrame(summary_rows)


def permutation_importance(frame: pd.DataFrame, coefficients: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    replicate_rows = []
    summary_rows = []
    z_columns = [f"{score}_Z" for score in SCORES]
    beta = coefficients.reindex(z_columns).to_numpy(dtype=float)
    for group in GROUPS:
        subset = frame.loc[frame["group"] == group].reset_index(drop=True)
        values = subset[z_columns].to_numpy(dtype=float)
        original_risk = values @ beta
        original_cindex = c_index(subset, original_risk)
        for score_index, score in enumerate(SCORES):
            deltas = np.empty(N_PERMUTATIONS, dtype=float)
            for replicate in range(N_PERMUTATIONS):
                permuted_values = values.copy()
                permuted_values[:, score_index] = rng.permutation(permuted_values[:, score_index])
                permuted_cindex = c_index(subset, permuted_values @ beta)
                delta = original_cindex - permuted_cindex
                deltas[replicate] = delta
                replicate_rows.append({
                    "group": group,
                    "score": score,
                    "display_name": DISPLAY_NAMES[score],
                    "replicate": replicate,
                    "original_cindex": original_cindex,
                    "permuted_cindex": permuted_cindex,
                    "delta_cindex": delta,
                })
            summary_rows.append({
                "group": group,
                "score": score,
                "display_name": DISPLAY_NAMES[score],
                "original_cindex": original_cindex,
                "mean_delta_cindex": float(np.mean(deltas)),
                "sd_delta_cindex": float(np.std(deltas, ddof=1)),
                "median_delta_cindex": float(np.median(deltas)),
                "percentile_2_5": float(np.quantile(deltas, 0.025)),
                "percentile_97_5": float(np.quantile(deltas, 0.975)),
                "proportion_delta_gt_0": float(np.mean(deltas > 0)),
            })
    summary = pd.DataFrame(summary_rows)
    summary["rank_within_group"] = summary.groupby("group")["mean_delta_cindex"].rank(method="first", ascending=False).astype(int)
    return pd.DataFrame(replicate_rows), summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_PATH)
    required = {"ID", "group", "duration", "event", *SCORES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    if frame["group"].value_counts().to_dict() != {"test2": 491, "train": 282, "test1": 113}:
        raise ValueError("Unexpected cohort sizes")
    if frame[list(required)].isna().any().any() or frame["ID"].duplicated().any():
        raise ValueError("Input has missing required values or duplicated IDs")

    standardized, scaler_metadata = standardize_from_train(frame)
    model, joint_summary = fit_joint_cox(standardized)
    correlation_matrix, correlation_pairs = correlation_outputs(standardized)
    vif = vif_output(standardized)
    patient_shap, shap_summary = analytic_shap(standardized, model.params_)
    permutation_replicates, permutation_summary = permutation_importance(standardized, model.params_)

    joint_summary.to_csv(OUTPUT_DIR / "five_score_joint_cox_train.csv", index=False, encoding="utf-8-sig")
    correlation_matrix.to_csv(OUTPUT_DIR / "five_score_pearson_correlation_train.csv", encoding="utf-8-sig")
    correlation_pairs.to_csv(OUTPUT_DIR / "five_score_correlation_pairs_train.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUTPUT_DIR / "five_score_vif_train.csv", index=False, encoding="utf-8-sig")
    patient_shap.to_csv(OUTPUT_DIR / "analytic_shap_patient_level.csv", index=False, encoding="utf-8-sig")
    shap_summary.to_csv(OUTPUT_DIR / "analytic_shap_summary.csv", index=False, encoding="utf-8-sig")
    permutation_replicates.to_csv(OUTPUT_DIR / "permutation_importance_replicates.csv", index=False, encoding="utf-8-sig")
    permutation_summary.to_csv(OUTPUT_DIR / "permutation_importance_summary.csv", index=False, encoding="utf-8-sig")

    habitat_correlations = correlation_pairs.loc[
        (correlation_pairs["score_1"] == "Habitat-max-H2") | (correlation_pairs["score_2"] == "Habitat-max-H2")
    ]
    habitat_cox = joint_summary.loc[joint_summary["score"] == "Habitat_Score"].iloc[0]
    habitat_vif = float(vif.loc[vif["score"] == "Habitat_Score", "vif"].iloc[0])
    manifest = {
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "fusion_manifest_sha256": sha256(FUSION_MANIFEST_PATH),
        "analysis_population": "Train for fitting/inference; Test1 and Test2 for fixed-model contribution descriptions only",
        "scores": SCORES,
        "standardization": "Train mean and population SD (ddof=0), applied unchanged to all cohorts",
        "joint_cox_penalizer": PENALIZER,
        "shap_definition": "Exact linear contribution beta_j * Z_ij; Train mean baseline is zero",
        "permutation_definition": "Fixed Train-fitted coefficients; within-cohort score permutation; original minus permuted Harrell C-index",
        "permutation_repeats": N_PERMUTATIONS,
        "seed": SEED,
        "habitat_key_results": {
            "joint_cox_beta": float(habitat_cox["coef"]),
            "joint_cox_hr": float(habitat_cox["exp(coef)"]),
            "joint_cox_ci_lower": float(habitat_cox["exp(coef) lower 95%"]),
            "joint_cox_ci_upper": float(habitat_cox["exp(coef) upper 95%"]),
            "joint_cox_p": float(habitat_cox["p"]),
            "vif": habitat_vif,
            "maximum_absolute_correlation_with_other_scores": float(habitat_correlations["absolute_r"].max()),
        },
        "interpretation_boundary": "Low correlation/VIF and a Train joint-Cox coefficient support distinct/non-redundant development-cohort information; they do not establish external incremental discrimination.",
    }
    (OUTPUT_DIR / "complementarity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )



def configure_cli():
    global INPUT_PATH, FUSION_MANIFEST_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train-based correlation/VIF and fixed-model linear contributions")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    INPUT_PATH, FUSION_MANIFEST_PATH, OUTPUT_DIR = args.scores.resolve(), args.manifest.resolve(), args.output.resolve()


if __name__ == "__main__":
    configure_cli()
    main()
