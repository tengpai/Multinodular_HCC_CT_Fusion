from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import socket
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis

RAW_PATH = None
LABEL_PATH = None
OUTPUT_DIR = None
SEED = 20260718
N_SPLITS = 5
MAX_CORRELATED_FEATURES = 2048
MAX_UNIVARIATE_FEATURES = 25
CORRELATION_THRESHOLD = 0.90
UNIVARIATE_P_THRESHOLD = 0.05
L1_RATIO = 0.5
FINAL_COX_PENALIZER = 0.1
ALPHA_RATIOS = np.logspace(0.0, -3.0, 13)
CANDIDATES = tuple((aggregation, region) for aggregation in ("mean", "min", "max") for region in ("H1", "H2", "H3", "Merged"))
EXPECTED_RAW_SHA256 = "ADA820B65ED7485A8D01518E15F2AC2ABA37629DED799BD89AB1FE97E1F521F4"
EXPECTED_LABEL_SHA256 = "A478782B8FC4A85C5AA13AF3C4CE026B0FBC0852C9BEE14F08B8446D27C85F10"
LOCK_FILE = None


@dataclass
class FitResult:
    state: dict[str, Any]
    linear_predictor: np.ndarray
    partial_hazard: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    for suffix in (".nii.gz", ".nii"):
        if normalized.lower().endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "scikit-survival", "lifelines"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def write_pickle(path: Path, value: Any) -> None:
    atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8-sig"))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def setup_logger() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("habitat_train_only")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / "run_habitat_train_only.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ProcessLock:
    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "stage": self.stage, "started_at": utc_now()})
        try:
            descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"Another run may be active; inspect {LOCK_FILE}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            LOCK_FILE.unlink()
        except FileNotFoundError:
            pass


def candidate_name(candidate: tuple[str, str]) -> str:
    return f"{candidate[0]}-{candidate[1]}"


def harrell_c(time: Iterable[float], event: Iterable[int], risk: Iterable[float]) -> float:
    time_values = np.asarray(list(time), dtype=float)
    event_values = np.asarray(list(event), dtype=int)
    risk_values = np.asarray(list(risk), dtype=float)
    if not np.isfinite(risk_values).all():
        raise ValueError("Non-finite risk prediction")
    return float(concordance_index(time_values, -risk_values, event_values))


def script_sha256() -> str:
    return sha256(Path(__file__).resolve())


def run_signature(raw_hash: str, label_hash: str) -> str:
    config = {
        "raw": raw_hash,
        "label": label_hash,
        "seed": SEED,
        "splits": N_SPLITS,
        "candidates": [candidate_name(c) for c in CANDIDATES],
        "alpha_ratios": ALPHA_RATIOS.tolist(),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "max_correlated_features": MAX_CORRELATED_FEATURES,
        "univariate_p": UNIVARIATE_P_THRESHOLD,
        "max_univariate": MAX_UNIVARIATE_FEATURES,
        "l1_ratio": L1_RATIO,
        "final_penalizer": FINAL_COX_PENALIZER,
        "script_sha256": script_sha256(),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest().upper()


def define_candidates(columns: list[str]) -> dict[str, Any]:
    identifiers = {"PatientID", "TumorLabel", "K_Value"}
    feature_columns = [column for column in columns if column not in identifiers]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate raw CSV column names")
    blocks = {region: [column for column in feature_columns if column.lower().endswith(f"_{region.lower()}")] for region in ("H1", "H2", "H3")}
    assigned = sum(blocks.values(), [])
    if set(assigned) != set(feature_columns) or len(assigned) != len(feature_columns):
        unassigned = sorted(set(feature_columns) - set(assigned))
        duplicates = sorted(column for column in set(assigned) if assigned.count(column) > 1)
        raise ValueError(f"Habitat feature partition is not exhaustive and exclusive; unassigned={unassigned[:10]}, duplicate={duplicates[:10]}")
    if any(len(blocks[region]) != 1834 for region in blocks):
        raise ValueError(f"Expected 1834 features per habitat, got { {key: len(value) for key, value in blocks.items()} }")
    merged = blocks["H1"] + blocks["H2"] + blocks["H3"]
    return {
        "candidate_order": [candidate_name(candidate) for candidate in CANDIDATES],
        "aggregation": ["mean", "min", "max"],
        "regions": ["H1", "H2", "H3", "Merged"],
        "feature_columns_by_region": {**blocks, "Merged": merged},
        "feature_count_by_region": {**{key: len(value) for key, value in blocks.items()}, "Merged": len(merged)},
        "excluded": ["volume_weighted", "worst_lesion", "outcome-derived lesion definitions"],
        "seed": SEED,
        "created_at": utc_now(),
    }


def load_cohort_identifiers() -> pd.DataFrame:
    cohorts = pd.read_csv(LABEL_PATH, usecols=["ID", "group"], dtype=str)
    cohorts["normalized_id"] = cohorts["ID"].map(normalize_id)
    cohorts["group"] = cohorts["group"].fillna("").str.strip()
    return cohorts


def load_train_endpoints(train_ids: set[str]) -> pd.DataFrame:
    endpoints = pd.read_csv(LABEL_PATH, usecols=["ID", "event", "duration"], dtype={"ID": str})
    endpoints["normalized_id"] = endpoints["ID"].map(normalize_id)
    endpoints = endpoints.loc[endpoints["normalized_id"].isin(train_ids), ["normalized_id", "event", "duration"]].copy()
    endpoints["event"] = pd.to_numeric(endpoints["event"], errors="raise").astype(int)
    endpoints["duration"] = pd.to_numeric(endpoints["duration"], errors="raise").astype(float)
    if not endpoints["event"].isin([0, 1]).all() or (endpoints["duration"] <= 0).any():
        raise ValueError("Invalid Train endpoint values")
    return endpoints


def audit_inputs(columns: list[str], identifiers: pd.DataFrame, cohorts: pd.DataFrame) -> dict[str, Any]:
    raw_ids = identifiers["PatientID"].map(normalize_id)
    cohort_ids = cohorts["normalized_id"]
    duplicate_labels = cohorts.loc[cohort_ids.duplicated(keep=False), ["ID", "group", "normalized_id"]]
    patient_tumors = pd.DataFrame({"normalized_id": raw_ids, "TumorLabel": identifiers["TumorLabel"].astype(str).str.strip()})
    duplicate_tumors = patient_tumors.loc[patient_tumors.duplicated(["normalized_id", "TumorLabel"], keep=False)]
    cross_queue = cohorts.groupby("normalized_id")["group"].nunique()
    unmatched_raw = sorted(set(raw_ids) - set(cohort_ids))
    unmatched_labels = sorted(set(cohort_ids) - set(raw_ids))
    k_values = pd.to_numeric(identifiers["K_Value"], errors="coerce")
    audit = {
        "raw_path": str(RAW_PATH),
        "raw_shape": [int(len(identifiers)), len(columns)],
        "raw_bytes": RAW_PATH.stat().st_size,
        "raw_sha256": sha256(RAW_PATH),
        "label_path": str(LABEL_PATH),
        "label_shape": [int(len(cohorts)), 4],
        "label_bytes": LABEL_PATH.stat().st_size,
        "label_sha256": sha256(LABEL_PATH),
        "patients": int(raw_ids.nunique()),
        "group_counts": {str(key): int(value) for key, value in cohorts["group"].value_counts(dropna=False).items()},
        "k_value_counts": {str(key): int(value) for key, value in k_values.value_counts(dropna=False).items()},
        "unmatched_raw_ids": unmatched_raw,
        "unmatched_label_ids": unmatched_labels,
        "duplicate_label_records": duplicate_labels.to_dict("records"),
        "duplicate_patient_tumor_records": duplicate_tumors.to_dict("records"),
        "cross_queue_duplicate_ids": sorted(cross_queue[cross_queue > 1].index.tolist()),
        "missing_patient_id_rows": int(raw_ids.eq("").sum()),
        "missing_tumor_label_rows": int(identifiers["TumorLabel"].isna().sum()),
    }
    write_json(OUTPUT_DIR / "raw_input_audit.json", audit)
    failures = [
        audit["raw_sha256"] != EXPECTED_RAW_SHA256,
        audit["label_sha256"] != EXPECTED_LABEL_SHA256,
        bool(unmatched_raw), bool(unmatched_labels), bool(duplicate_labels.shape[0]),
        bool(duplicate_tumors.shape[0]), bool(audit["cross_queue_duplicate_ids"]),
        audit["missing_patient_id_rows"] > 0, audit["missing_tumor_label_rows"] > 0,
        not k_values.notna().all(), set(k_values.dropna().unique()) != {3},
    ]
    if any(failures):
        raise ValueError("Frozen-input audit failed; inspect raw_input_audit.json")
    return audit


def fixed_folds(labels: pd.DataFrame, seed: int) -> np.ndarray:
    folds = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (_, valid_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels["event"].astype(int))):
        folds[valid_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("Incomplete fixed fold assignment")
    return folds


def aggregate_and_save(raw: pd.DataFrame, definition: dict[str, Any], logger: logging.Logger) -> None:
    raw["normalized_id"] = raw["PatientID"].map(normalize_id)
    all_features = definition["feature_columns_by_region"]["Merged"]
    numeric = raw.loc[:, all_features].apply(pd.to_numeric, errors="coerce")
    numeric.insert(0, "normalized_id", raw["normalized_id"].to_numpy())
    grouped_source = numeric.groupby("normalized_id", sort=True)
    for aggregation in ("mean", "min", "max"):
        grouped = getattr(grouped_source, aggregation)().astype(np.float32)
        for region in ("H1", "H2", "H3", "Merged"):
            name = f"{aggregation}-{region}"
            candidate = grouped.loc[:, definition["feature_columns_by_region"][region]]
            candidate.index.name = "normalized_id"
            write_pickle(OUTPUT_DIR / f"candidate_{name}.pkl", candidate)
            logger.info("prepared %s: %d patients x %d features", name, candidate.shape[0], candidate.shape[1])
        del grouped


def prepare(logger: logging.Logger) -> None:
    started = utc_now()
    columns = pd.read_csv(RAW_PATH, nrows=0).columns.tolist()
    definition = define_candidates(columns)
    cohorts = load_cohort_identifiers()
    identifiers = pd.read_csv(RAW_PATH, usecols=["PatientID", "TumorLabel", "K_Value"], dtype={"PatientID": str})
    audit = audit_inputs(columns, identifiers, cohorts)
    write_json(OUTPUT_DIR / "candidate_definition.json", definition)
    train_ids = set(cohorts.loc[cohorts["group"].str.lower() == "train", "normalized_id"])
    train_labels = load_train_endpoints(train_ids).sort_values("normalized_id").reset_index(drop=True)
    if len(train_labels) != 282 or set(train_labels["normalized_id"]) != train_ids:
        raise ValueError("Train endpoint isolation produced an unexpected cohort")
    fold_frame = train_labels.loc[:, ["normalized_id"]].copy()
    fold_frame["fold"] = fixed_folds(train_labels, SEED)
    write_csv(OUTPUT_DIR / "fold_assignments.csv", fold_frame)
    write_csv(OUTPUT_DIR / "train_labels.csv", train_labels)
    dtype_map = {column: np.float32 for column in definition["feature_columns_by_region"]["Merged"]}
    dtype_map.update({"PatientID": str, "TumorLabel": str, "K_Value": np.float32})
    logger.info("reading frozen raw feature CSV once using float32 feature storage")
    raw = pd.read_csv(RAW_PATH, dtype=dtype_map)
    aggregate_and_save(raw, definition, logger)
    manifest = {
        "stage": "prepare", "status": "complete", "started_at": started, "finished_at": utc_now(),
        "raw_sha256": audit["raw_sha256"], "label_sha256": audit["label_sha256"],
        "script_sha256": script_sha256(), "run_signature": run_signature(audit["raw_sha256"], audit["label_sha256"]),
        "python": sys.version, "platform": platform.platform(), "versions": package_versions(),
        "external_endpoint_policy": "Only ID/group define cohorts; event/duration are immediately restricted to the frozen Train ID whitelist. No external endpoint is persisted or evaluated.",
    }
    write_json(OUTPUT_DIR / "prepare_manifest.json", manifest)
    logger.info("prepare complete")


def survival_array(labels: pd.DataFrame) -> np.ndarray:
    return np.array(list(zip(labels["event"].astype(bool), labels["duration"].astype(float))), dtype=[("event", "?"), ("duration", "<f8")])


def fit_preprocessor(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = features.reset_index(drop=True).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = labels.reset_index(drop=True)
    available = features.notna().any(axis=0) & (features.nunique(dropna=True) > 1)
    features = features.loc[:, available]
    medians = features.median().dropna()
    features = features.loc[:, medians.index].fillna(medians)
    means = features.mean()
    scales = features.std(ddof=0)
    finite_scale = scales.notna() & np.isfinite(scales) & (scales > 0)
    features = features.loc[:, finite_scale]
    medians, means, scales = medians.loc[features.columns], means.loc[features.columns], scales.loc[features.columns]
    standardized = (features - means) / scales
    correlated_columns = correlation_filter(standardized)
    standardized = standardized.loc[:, correlated_columns]
    univariate = univariate_filter(standardized, labels)
    selected = [record["feature"] for record in univariate if record["p"] < UNIVARIATE_P_THRESHOLD][:MAX_UNIVARIATE_FEATURES]
    if len(selected) < 2:
        raise ValueError(f"univariate_cox selected fewer than two features ({len(selected)})")
    state = {
        "available_columns": features.columns.tolist(), "medians": medians.to_dict(), "means": means.to_dict(),
        "scales": scales.to_dict(), "correlation_columns": correlated_columns, "univariate_results": univariate,
        "univariate_columns": selected,
    }
    return standardized.loc[:, selected], state


def transform_preprocessor(features: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    transformed = features.reset_index(drop=True).reindex(columns=state["available_columns"])
    transformed = transformed.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    transformed = transformed.fillna(pd.Series(state["medians"]))
    transformed = (transformed - pd.Series(state["means"])) / pd.Series(state["scales"])
    transformed = transformed.reindex(columns=state["univariate_columns"])
    if transformed.isna().any().any() or not np.isfinite(transformed.to_numpy()).all():
        raise ValueError("Validation transform contains non-finite values after training-fold preprocessing")
    return transformed


def correlation_filter(frame: pd.DataFrame) -> list[str]:
    values = np.asarray(frame.to_numpy(copy=False), dtype=np.float32, order="C")
    correlation = (values.T @ values) / np.float32(max(values.shape[0], 1))
    blocked = np.zeros(values.shape[1], dtype=bool)
    keep: list[int] = []
    for index in range(values.shape[1]):
        if blocked[index]:
            continue
        keep.append(index)
        if len(keep) >= MAX_CORRELATED_FEATURES:
            break
        later = np.abs(correlation[index, index + 1 :]) > CORRELATION_THRESHOLD
        blocked[index + 1 :] |= later
    return [frame.columns[index] for index in keep]


def univariate_filter(features: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, Any]]:
    labels = labels.reset_index(drop=True)
    order = np.argsort(-labels["duration"].to_numpy(dtype=float), kind="stable")
    times = labels["duration"].to_numpy(dtype=float)[order]
    events = labels["event"].to_numpy(dtype=int)[order]
    values = features.to_numpy(dtype=float, copy=False)[order]
    cumulative_x = np.cumsum(values, axis=0)
    cumulative_x2 = np.cumsum(values * values, axis=0)
    score = np.zeros(values.shape[1], dtype=float)
    information = np.zeros(values.shape[1], dtype=float)
    for time_value in np.unique(times[events == 1]):
        tied = times == time_value
        event_mask = tied & (events == 1)
        event_count = int(event_mask.sum())
        risk_end = int(np.flatnonzero(tied)[-1])
        risk_count = risk_end + 1
        risk_mean = cumulative_x[risk_end] / risk_count
        risk_variance = cumulative_x2[risk_end] / risk_count - risk_mean * risk_mean
        score += values[event_mask].sum(axis=0) - event_count * risk_mean
        information += event_count * np.maximum(risk_variance, 0.0)
    statistics = np.divide(score * score, information, out=np.zeros_like(score), where=information > 0)
    from scipy.stats import chi2
    p_values = chi2.sf(statistics, 1)
    results = [
        {"feature": column, "p": float(p_value), "status": "complete", "test": "univariate Cox score test"}
        for column, p_value in zip(features.columns, p_values)
    ]
    return sorted(results, key=lambda record: (record["p"], record["feature"]))


def alpha_max_from_training(features: pd.DataFrame, labels: pd.DataFrame) -> float:
    model = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, n_alphas=30, alpha_min_ratio=0.001, max_iter=10000)
    model.fit(features.to_numpy(dtype=float), survival_array(labels))
    alpha_max = float(np.max(model.alphas_))
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        raise ValueError("Could not derive a positive finite alpha_max from the training fold")
    return alpha_max


def fit_at_ratio(train_features: pd.DataFrame, train_labels: pd.DataFrame, valid_features: pd.DataFrame, alpha_ratio: float) -> FitResult:
    train_labels = train_labels.reset_index(drop=True)
    transformed_train, state = fit_preprocessor(train_features, train_labels)
    transformed_valid = transform_preprocessor(valid_features, state)
    alpha_max = alpha_max_from_training(transformed_train, train_labels)
    alpha = float(alpha_max * alpha_ratio)
    elastic_net = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=[alpha], max_iter=10000)
    elastic_net.fit(transformed_train.to_numpy(dtype=float), survival_array(train_labels))
    coefficients = np.asarray(elastic_net.coef_).reshape(-1)
    nonzero_indices = np.flatnonzero(np.abs(coefficients) > 1e-10)
    if len(nonzero_indices) < 1:
        raise ValueError("Elastic Net selected no feature")
    nonzero_features = [transformed_train.columns[index] for index in nonzero_indices]
    cox_train = transformed_train.loc[:, nonzero_features].copy()
    cox_train["event"] = train_labels["event"].to_numpy()
    cox_train["duration"] = train_labels["duration"].to_numpy()
    final_cox = CoxPHFitter(penalizer=FINAL_COX_PENALIZER)
    final_cox.fit(cox_train, duration_col="duration", event_col="event")
    valid_schema = transformed_valid.reindex(columns=nonzero_features)
    linear_predictor = final_cox.predict_log_partial_hazard(valid_schema).to_numpy().reshape(-1)
    partial_hazard = final_cox.predict_partial_hazard(valid_schema).to_numpy().reshape(-1)
    state.update({
        "alpha_ratio": float(alpha_ratio), "alpha_max": alpha_max, "alpha": alpha,
        "elastic_net_coefficients": dict(zip(transformed_train.columns, coefficients.tolist())),
        "nonzero_features": nonzero_features, "cox_coefficients": final_cox.params_.to_dict(), "cox_model": final_cox,
    })
    return FitResult(state=state, linear_predictor=linear_predictor, partial_hazard=partial_hazard)


def tune_candidate(features: pd.DataFrame, labels: pd.DataFrame, folds: np.ndarray, context: str) -> dict[str, Any]:
    predictions = {float(ratio): np.full(len(labels), np.nan) for ratio in ALPHA_RATIOS}
    fold_details: list[dict[str, Any]] = []
    for fold in range(N_SPLITS):
        fit_indices = np.flatnonzero(folds != fold)
        valid_indices = np.flatnonzero(folds == fold)
        for ratio in ALPHA_RATIOS:
            try:
                fitted = fit_at_ratio(features.iloc[fit_indices], labels.iloc[fit_indices], features.iloc[valid_indices], float(ratio))
                predictions[float(ratio)][valid_indices] = fitted.linear_predictor
                fold_details.append({"context": context, "fold": fold, "alpha_ratio": float(ratio), "status": "complete", "n_features": len(fitted.state["nonzero_features"])})
            except Exception as error:
                fold_details.append({"context": context, "fold": fold, "alpha_ratio": float(ratio), "status": "failed", "step": "fold_pipeline", "error": repr(error), "traceback": traceback.format_exc()})
    ratio_rows: list[dict[str, Any]] = []
    for ratio in ALPHA_RATIOS:
        risk = predictions[float(ratio)]
        complete = np.isfinite(risk).all()
        score = harrell_c(labels["duration"], labels["event"], risk) if complete else np.nan
        ratio_rows.append({"alpha_ratio": float(ratio), "status": "complete" if complete else "failed", "oof_cindex": score})
    eligible = [row for row in ratio_rows if row["status"] == "complete"]
    if not eligible:
        raise RuntimeError(f"All alpha ratios failed for {context}")
    best = sorted(eligible, key=lambda row: (-row["oof_cindex"], -row["alpha_ratio"]))[0]
    best_ratio = float(best["alpha_ratio"])
    best_risk = predictions[best_ratio]
    fold_scores = [harrell_c(labels.iloc[folds == fold]["duration"], labels.iloc[folds == fold]["event"], best_risk[folds == fold]) for fold in range(N_SPLITS)]
    feature_counts = [row["n_features"] for row in fold_details if row["status"] == "complete" and row["alpha_ratio"] == best_ratio]
    return {
        "status": "complete", "context": context, "best_alpha_ratio": best_ratio,
        "oof_cindex": float(best["oof_cindex"]), "fold_cindex_mean": float(np.mean(fold_scores)),
        "fold_cindex_sd": float(np.std(fold_scores, ddof=1)), "mean_n_features": float(np.mean(feature_counts)),
        "oof_linear_predictor": best_risk, "ratio_results": ratio_rows, "fold_details": fold_details,
    }


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("status") == "complete" and np.isfinite(row.get("oof_cindex", np.nan))]
    if not eligible:
        raise RuntimeError("No candidate completed")
    highest = max(row["oof_cindex"] for row in eligible)
    near = [row for row in eligible if highest - row["oof_cindex"] <= 0.005]
    order = {candidate_name(candidate): index for index, candidate in enumerate(CANDIDATES)}
    return sorted(near, key=lambda row: (row["mean_n_features"], row["fold_cindex_sd"], order[row["candidate"]]))[0]


def candidate_frame(name: str, train_labels: pd.DataFrame) -> pd.DataFrame:
    frame = read_pickle(OUTPUT_DIR / f"candidate_{name}.pkl")
    requested_ids = train_labels["normalized_id"].astype(str).tolist()
    missing_ids = sorted(set(requested_ids) - set(frame.index.astype(str)))
    if missing_ids:
        raise ValueError(f"Candidate {name} is missing Train patients: {missing_ids[:10]}")
    aligned = frame.reindex(requested_ids)
    if len(aligned) != len(train_labels):
        raise ValueError(f"Candidate {name} cannot align to all Train patients")
    return aligned.reset_index(drop=True)


def checkpoint_path(kind: str, context: str) -> Path:
    safe_context = context.replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / "checkpoints" / f"{kind}_{safe_context}.pkl"


def load_checkpoint(path: Path, signature: str) -> Any | None:
    if not path.exists():
        return None
    value = read_pickle(path)
    return value.get("payload") if value.get("run_signature") == signature and value.get("status") == "complete" else None


def save_checkpoint(path: Path, signature: str, payload: Any) -> None:
    write_pickle(path, {"status": "complete", "run_signature": signature, "saved_at": utc_now(), "payload": payload})


def run_candidate_oof(train_labels: pd.DataFrame, fixed: np.ndarray, signature: str, logger: logging.Logger) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    all_fold_details: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        name = candidate_name(candidate)
        path = checkpoint_path("candidate_oof", name)
        result = load_checkpoint(path, signature)
        if result is None:
            try:
                result = tune_candidate(candidate_frame(name, train_labels), train_labels, fixed, f"final-selection/{name}")
                result["candidate"] = name
                save_checkpoint(path, signature, result)
            except Exception as error:
                result = {"candidate": name, "status": "failed", "step": "candidate_oof", "error": repr(error), "traceback": traceback.format_exc()}
                logger.error("candidate OOF failed: %s: %s", name, error)
        summaries.append({key: value for key, value in result.items() if key not in {"oof_linear_predictor", "ratio_results", "fold_details"}})
        if result.get("status") == "complete":
            all_fold_details.extend({"candidate": name, **row} for row in result["fold_details"])
            for index, risk in enumerate(result["oof_linear_predictor"]):
                prediction_rows.append({"normalized_id": train_labels.iloc[index]["normalized_id"], "candidate": name, "fold": int(fixed[index]), "linear_predictor": float(risk)})
    return summaries, pd.DataFrame(prediction_rows), all_fold_details


def run_nested_cv(train_labels: pd.DataFrame, fixed: np.ndarray, signature: str, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    frames = {candidate_name(candidate): candidate_frame(candidate_name(candidate), train_labels) for candidate in CANDIDATES}
    for outer_fold in range(N_SPLITS):
        path = checkpoint_path("nested_outer", str(outer_fold))
        result = load_checkpoint(path, signature)
        if result is None:
            fit_indices = np.flatnonzero(fixed != outer_fold)
            valid_indices = np.flatnonzero(fixed == outer_fold)
            outer_labels = train_labels.iloc[fit_indices].reset_index(drop=True)
            inner_folds = fixed_folds(outer_labels, SEED + 1000 + outer_fold)
            candidate_results: list[dict[str, Any]] = []
            for candidate in CANDIDATES:
                name = candidate_name(candidate)
                try:
                    tuned = tune_candidate(frames[name].iloc[fit_indices].reset_index(drop=True), outer_labels, inner_folds, f"nested/outer-{outer_fold}/{name}")
                    candidate_results.append({"candidate": name, **{key: value for key, value in tuned.items() if key not in {"oof_linear_predictor", "ratio_results", "fold_details"}}})
                except Exception as error:
                    candidate_results.append({"candidate": name, "status": "failed", "step": "inner_candidate_selection", "error": repr(error)})
            chosen = choose_candidate(candidate_results)
            fitted = fit_at_ratio(frames[chosen["candidate"]].iloc[fit_indices], outer_labels, frames[chosen["candidate"]].iloc[valid_indices], chosen["best_alpha_ratio"])
            result = {
                "outer_fold": outer_fold, "status": "complete", "selected_candidate": chosen["candidate"],
                "selected_alpha_ratio": chosen["best_alpha_ratio"], "selected_inner_oof_cindex": chosen["oof_cindex"],
                "n_features": len(fitted.state["nonzero_features"]), "features": fitted.state["nonzero_features"],
                "valid_indices": valid_indices, "linear_predictor": fitted.linear_predictor,
                "partial_hazard": fitted.partial_hazard, "candidate_results": candidate_results,
            }
            save_checkpoint(path, signature, result)
        outer_risk = np.asarray(result["linear_predictor"], dtype=float)
        valid_indices = np.asarray(result["valid_indices"], dtype=int)
        outer_cindex = harrell_c(train_labels.iloc[valid_indices]["duration"], train_labels.iloc[valid_indices]["event"], outer_risk)
        outer_rows.append({
            "outer_fold": outer_fold, "status": result["status"], "selected_candidate": result["selected_candidate"],
            "selected_alpha_ratio": result["selected_alpha_ratio"], "selected_inner_oof_cindex": result["selected_inner_oof_cindex"],
            "outer_cindex": outer_cindex, "n_features": result["n_features"], "selected_features": "|".join(result["features"]),
        })
        for position, index in enumerate(valid_indices):
            prediction_rows.append({"normalized_id": train_labels.iloc[index]["normalized_id"], "outer_fold": outer_fold, "selected_candidate": result["selected_candidate"], "linear_predictor": float(outer_risk[position]), "partial_hazard": float(result["partial_hazard"][position])})
    return pd.DataFrame(outer_rows), pd.DataFrame(prediction_rows)


def jaccard_rows(feature_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    names = sorted(feature_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            union = feature_sets[left] | feature_sets[right]
            rows.append({"left": left, "right": right, "jaccard": len(feature_sets[left] & feature_sets[right]) / len(union) if union else np.nan})
    return pd.DataFrame(rows)


def artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def select(logger: logging.Logger) -> None:
    started = utc_now()
    required_prepare = ["candidate_definition.json", "fold_assignments.csv", "train_labels.csv", "prepare_manifest.json"]
    missing = [name for name in required_prepare if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run --stage prepare first; missing {missing}")
    prepare_manifest = json.loads((OUTPUT_DIR / "prepare_manifest.json").read_text(encoding="utf-8"))
    signature = run_signature(prepare_manifest["raw_sha256"], prepare_manifest["label_sha256"])
    if signature != prepare_manifest["run_signature"]:
        raise RuntimeError("Prepare outputs were generated by a different script/configuration; rerun prepare")
    train_labels = pd.read_csv(OUTPUT_DIR / "train_labels.csv")
    folds_frame = pd.read_csv(OUTPUT_DIR / "fold_assignments.csv")
    if list(train_labels["normalized_id"]) != list(folds_frame["normalized_id"]):
        raise ValueError("Train labels and fixed folds are misaligned")
    fixed = folds_frame["fold"].to_numpy(dtype=int)
    summaries, oof_predictions, fold_details = run_candidate_oof(train_labels, fixed, signature, logger)
    summary_frame = pd.DataFrame(summaries)
    write_csv(OUTPUT_DIR / "candidate_train_oof_summary.csv", summary_frame)
    write_csv(OUTPUT_DIR / "candidate_train_oof_predictions.csv", oof_predictions)
    write_csv(OUTPUT_DIR / "candidate_train_oof_fold_details.csv", pd.DataFrame(fold_details))
    chosen = choose_candidate(summaries)
    selected = {
        "candidate": chosen["candidate"], "best_alpha_ratio": chosen["best_alpha_ratio"],
        "oof_cindex": chosen["oof_cindex"], "mean_n_features": chosen["mean_n_features"],
        "fold_cindex_sd": chosen["fold_cindex_sd"],
        "rule": "Highest pooled Train OOF Harrell C-index; candidates within 0.005 use fewer mean fold-selected features, lower fold C-index SD, then frozen order.",
        "locked_at": utc_now(), "run_signature": signature,
    }
    nested_summary, nested_predictions = run_nested_cv(train_labels, fixed, signature, logger)
    write_csv(OUTPUT_DIR / "nested_cv_summary.csv", nested_summary)
    write_csv(OUTPUT_DIR / "nested_cv_predictions.csv", nested_predictions)
    nested_pooled = harrell_c(train_labels.set_index("normalized_id").loc[nested_predictions["normalized_id"], "duration"], train_labels.set_index("normalized_id").loc[nested_predictions["normalized_id"], "event"], nested_predictions["linear_predictor"])
    selected["nested_cv_pooled_cindex"] = nested_pooled
    selected["nested_candidate_frequency"] = nested_summary["selected_candidate"].value_counts().to_dict()
    write_json(OUTPUT_DIR / "selected_candidate.json", selected)
    features = candidate_frame(chosen["candidate"], train_labels)
    final_fit = fit_at_ratio(features, train_labels, features, chosen["best_alpha_ratio"])
    final_fit.state.update({
        "candidate": chosen["candidate"], "candidate_order": [candidate_name(candidate) for candidate in CANDIDATES],
        "habitat_score_definition": "final Cox linear predictor (log partial hazard)",
        "cutoff_train_median": float(np.median(final_fit.linear_predictor)), "versions": package_versions(),
        "run_signature": signature, "raw_sha256": prepare_manifest["raw_sha256"], "label_sha256": prepare_manifest["label_sha256"],
        "script_sha256": script_sha256(), "fitted_at": utc_now(),
    })
    write_pickle(OUTPUT_DIR / "final_habitat_model.pkl", final_fit.state)
    feature_sets = {f"outer-{int(row.outer_fold)}": set(str(row.selected_features).split("|")) for row in nested_summary.itertuples()}
    write_csv(OUTPUT_DIR / "nested_feature_jaccard.csv", jaccard_rows(feature_sets))
    gate_names = [
        "candidate_definition.json", "fold_assignments.csv", "nested_cv_summary.csv",
        "candidate_train_oof_summary.csv", "selected_candidate.json", "final_habitat_model.pkl",
    ]
    artifacts = [artifact_record(OUTPUT_DIR / name) for name in gate_names]
    manifest = {
        "status": "locked", "started_at": started, "locked_at": utc_now(), "candidate": chosen["candidate"],
        "run_signature": signature, "raw_sha256": prepare_manifest["raw_sha256"], "label_sha256": prepare_manifest["label_sha256"],
        "script_sha256": script_sha256(), "versions": package_versions(), "artifacts": artifacts,
        "test1_gate": "A future Test 1 evaluator must verify every artifact hash in this manifest before reading any Test 1 endpoint.",
    }
    write_json(OUTPUT_DIR / "model_lock_manifest.json", manifest)
    logger.info("select complete; model locked using Train only; Test 1 outcomes were not read or evaluated by this stage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable Train-only Habitat re-analysis")
    parser.add_argument("--stage", required=True, choices=("prepare", "select"))
    arguments = parser.parse_args()
    logger = setup_logger()
    with ProcessLock(arguments.stage):
        logger.info("stage=%s started", arguments.stage)
        try:
            prepare(logger) if arguments.stage == "prepare" else select(logger)
        except Exception:
            failure = {"stage": arguments.stage, "status": "failed", "failed_at": utc_now(), "traceback": traceback.format_exc(), "script_sha256": script_sha256()}
            write_json(OUTPUT_DIR / f"{arguments.stage}_failure.json", failure)
            logger.exception("stage=%s failed", arguments.stage)
            raise



def configure_cli():
    global RAW_PATH, LABEL_PATH, OUTPUT_DIR, LOCK_FILE
    global EXPECTED_RAW_SHA256, EXPECTED_LABEL_SHA256
    parser = argparse.ArgumentParser(description="Portable Train-only 12-candidate Habitat analysis")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "select"), required=True)
    parser.add_argument("--expected-raw-sha256", default=EXPECTED_RAW_SHA256)
    parser.add_argument("--expected-label-sha256", default=EXPECTED_LABEL_SHA256)
    args = parser.parse_args()
    RAW_PATH, LABEL_PATH, OUTPUT_DIR = args.raw.resolve(), args.labels.resolve(), args.output.resolve()
    LOCK_FILE = OUTPUT_DIR / "run_habitat_train_only.lock"
    EXPECTED_RAW_SHA256 = args.expected_raw_sha256.upper()
    EXPECTED_LABEL_SHA256 = args.expected_label_sha256.upper()
    # The original main keeps its original stage parser and locking behavior.
    sys.argv = [sys.argv[0], "--stage", args.stage]


if __name__ == "__main__":
    configure_cli()
    main()
