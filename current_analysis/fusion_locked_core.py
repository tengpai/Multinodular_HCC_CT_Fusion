from __future__ import annotations
import argparse

import hashlib
import importlib.metadata
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = None
HABITAT_DIR = None
HABITAT_MANIFEST = None
HABITAT_MODEL = None
HABITAT_CANDIDATE = None
FUSION_INPUT = None
OUTPUT_DIR = None
CLINICAL_VARS = ["Child_Pugh", "Gender", "Total_SLD_cm", "Age", "Diabetes", "AFP", "CA199", "Lesion_Count", "Hepatitis_C"]
SCORE_COLUMNS = ["Clinical_Score", "Rad_Diff_HighRisk", "DL_Heterogeneity", "DL_MaxPooling", "Habitat_Score"]
MODEL_DEFINITIONS = {
    "Clinical-only": ["Clinical_Score"],
    "Clinical_RadDiff_DLHet": ["Clinical_Score", "Rad_Diff_HighRisk", "DL_Heterogeneity"],
    "Four-score_without_Habitat": ["Clinical_Score", "Rad_Diff_HighRisk", "DL_Heterogeneity", "DL_MaxPooling"],
    "Full_five-score_Fusion": ["Clinical_Score", "Rad_Diff_HighRisk", "DL_Heterogeneity", "DL_MaxPooling", "Habitat_Score"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def package_versions() -> dict[str, str]:
    result = {}
    for package in ("numpy", "pandas", "lifelines", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def c_index(frame: pd.DataFrame, score_column: str) -> float:
    return float(concordance_index(frame["duration"], -frame[score_column], frame["event"]))


def verify_habitat_lock() -> dict:
    if not HABITAT_MANIFEST.exists() or not HABITAT_MODEL.exists() or not HABITAT_CANDIDATE.exists():
        raise FileNotFoundError("Habitat lock gate is incomplete")
    manifest = json.loads(HABITAT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "locked":
        raise RuntimeError("Habitat model lock status is not locked")
    for artifact in manifest["artifacts"]:
        path = HABITAT_DIR / artifact["path"]
        if not path.exists() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Habitat artifact hash verification failed: {path}")
    if manifest.get("candidate") != "max-H2":
        raise RuntimeError("Locked Habitat candidate is not max-H2")
    return manifest


def transform_habitat(features: pd.DataFrame, state: dict) -> pd.DataFrame:
    transformed = features.reindex(columns=state["available_columns"])
    transformed = transformed.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    transformed = transformed.fillna(pd.Series(state["medians"]))
    transformed = (transformed - pd.Series(state["means"])) / pd.Series(state["scales"])
    transformed = transformed.reindex(columns=state["nonzero_features"])
    if transformed.isna().any().any() or not np.isfinite(transformed.to_numpy()).all():
        raise ValueError("Habitat transform contains non-finite values")
    return transformed


def build_clinical_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = frame.loc[frame["group"] == "train"].copy()
    scaler = StandardScaler()
    scaler.fit(train[CLINICAL_VARS])
    all_scaled = scaler.transform(frame[CLINICAL_VARS])
    coefficients = {}
    univariate = []
    for index, variable in enumerate(CLINICAL_VARS):
        fit_frame = train[["duration", "event", variable]].dropna().copy()
        model = CoxPHFitter(penalizer=0.1)
        model.fit(fit_frame, duration_col="duration", event_col="event")
        coefficients[variable] = float(model.params_[variable])
        univariate.append({"variable": variable, "coefficient": float(model.params_[variable]), "p": float(model.summary.loc[variable, "p"])})
    score = np.zeros(len(frame), dtype=float)
    for index, variable in enumerate(CLINICAL_VARS):
        score += coefficients[variable] * all_scaled[:, index]
    result = frame.copy()
    result["Clinical_Score"] = score
    return result, {"clinical_variables": CLINICAL_VARS, "coefficients": coefficients, "univariate": univariate, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}


def fit_model(frame: pd.DataFrame, model_name: str, predictors: list[str]) -> tuple[dict, pd.DataFrame]:
    train = frame.loc[frame["group"] == "train"].copy()
    scalers = {}
    z_columns = []
    for predictor in predictors:
        scaler = StandardScaler()
        train[f"{predictor}_Z"] = scaler.fit_transform(train[[predictor]])
        frame[f"{predictor}_Z"] = scaler.transform(frame[[predictor]])
        scalers[predictor] = {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])}
        z_columns.append(f"{predictor}_Z")
    fit_frame = train[["duration", "event"] + z_columns].dropna()
    model = CoxPHFitter(penalizer=0.01)
    model.fit(fit_frame, duration_col="duration", event_col="event")
    frame[f"{model_name}_Score"] = model.predict_log_partial_hazard(frame[z_columns]).to_numpy().reshape(-1)
    return {"model_name": model_name, "predictors": predictors, "scalers": scalers, "coefficients": {column: float(model.params_[column]) for column in z_columns}, "n_train": int(len(fit_frame)), "penalizer": 0.01}, frame


def main() -> None:
    started = utc_now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = verify_habitat_lock()
    source_hash = sha256(FUSION_INPUT)
    frame = pd.read_csv(FUSION_INPUT)
    expected_columns = {"ID", "group", "event", "duration", *CLINICAL_VARS, "Rad_Diff_HighRisk", "DL_Heterogeneity", "DL_MaxPooling"}
    if not expected_columns.issubset(frame.columns):
        raise ValueError("Fusion input is missing required columns")
    if len(frame) != 886 or frame["ID"].duplicated().any():
        raise ValueError("Fusion input cohort structure failed")
    if frame["group"].value_counts().to_dict() != {"test2": 491, "train": 282, "test1": 113}:
        raise ValueError("Unexpected cohort sizes")
    frame["event"] = pd.to_numeric(frame["event"], errors="raise").astype(int)
    frame["duration"] = pd.to_numeric(frame["duration"], errors="raise").astype(float)
    habitat_state = pickle.loads(HABITAT_MODEL.read_bytes())
    habitat_candidate = pickle.loads(HABITAT_CANDIDATE.read_bytes())
    habitat_features = habitat_candidate.reindex(frame["ID"].astype(str).str.removesuffix(".nii.gz"))
    if habitat_features.isna().all(axis=1).any():
        habitat_features = habitat_candidate.reindex(frame["ID"])
    if len(habitat_features) != len(frame) or habitat_features.isna().all(axis=1).any():
        raise ValueError("Habitat candidate cannot align to fusion IDs")
    transformed_habitat = transform_habitat(habitat_features, habitat_state)
    frame["Habitat_Score"] = habitat_state["cox_model"].predict_log_partial_hazard(transformed_habitat).to_numpy().reshape(-1)
    frame, clinical_metadata = build_clinical_scores(frame)
    expected_legacy_cindex = {"train": 0.6282491000311801, "test1": 0.5853217642805495, "test2": 0.5918735458910257}
    reproduced_legacy_cindex = {group: c_index(frame.loc[frame["group"] == group], "Clinical_Score") for group in expected_legacy_cindex}
    if any(abs(reproduced_legacy_cindex[group] - expected_legacy_cindex[group]) > 1e-12 for group in expected_legacy_cindex):
        raise RuntimeError(f"Original Clinical Score reproduction failed: {reproduced_legacy_cindex}")
    clinical_metadata["reproduced_legacy_cindex"] = reproduced_legacy_cindex
    clinical_metadata["expected_legacy_cindex"] = expected_legacy_cindex
    model_metadata = {}
    result_rows = []
    for model_name, predictors in MODEL_DEFINITIONS.items():
        metadata, frame = fit_model(frame, model_name, predictors)
        model_metadata[model_name] = metadata
        score_column = f"{model_name}_Score"
        for group in ("train", "test1", "test2"):
            subset = frame.loc[frame["group"] == group]
            result_rows.append({"model": model_name, "group": group, "endpoint": "PFS exploratory migration" if group == "test2" else "RFS", "n": int(len(subset)), "events": int(subset["event"].sum()), "c_index": c_index(subset, score_column)})
    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(OUTPUT_DIR / "fusion_core_cindex.csv", index=False, encoding="utf-8-sig")
    frame[["ID", "group", "event", "duration"] + SCORE_COLUMNS + [f"{name}_Score" for name in MODEL_DEFINITIONS]].to_csv(OUTPUT_DIR / "fusion_scores_locked.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "fusion_core_manifest.json", {"status": "complete", "started_at": started, "finished_at": utc_now(), "fusion_input": str(FUSION_INPUT), "fusion_input_sha256": source_hash, "habitat_lock_manifest_sha256": sha256(HABITAT_MANIFEST), "habitat_candidate": manifest["candidate"], "models": MODEL_DEFINITIONS, "clinical_metadata": clinical_metadata, "model_metadata": model_metadata, "packages": package_versions(), "test1_policy": "Test1 evaluated once after verifying the Habitat lock; no Test1 result participates in model or cutoff selection.", "test2_policy": "Test2 is exploratory PFS migration and is not averaged with Test1 or used for selection."})
    print(result_frame.to_string(index=False))



def configure_cli():
    global HABITAT_DIR, HABITAT_MANIFEST, HABITAT_MODEL, HABITAT_CANDIDATE, FUSION_INPUT, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Locked Habitat gate followed by Train-only fusion fitting")
    parser.add_argument("--habitat-dir", type=Path, required=True)
    parser.add_argument("--fusion-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    HABITAT_DIR = args.habitat_dir.resolve()
    HABITAT_MANIFEST = HABITAT_DIR / "model_lock_manifest.json"
    HABITAT_MODEL = HABITAT_DIR / "final_habitat_model.pkl"
    HABITAT_CANDIDATE = HABITAT_DIR / "candidate_max-H2.pkl"
    FUSION_INPUT, OUTPUT_DIR = args.fusion_input.resolve(), args.output.resolve()


if __name__ == "__main__":
    configure_cli()
    main()
