from __future__ import annotations
import argparse
INPUT_FILE = None
CLINICAL_INPUT = None

import json
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

PROJECT_DIR = None
INPUT_DIR = None
OUTPUT_DIR = None
SCORES = ["Clinical-only_Score", "Clinical_RadDiff_DLHet_Score", "Four-score_without_Habitat_Score", "Full_five-score_Fusion_Score"]
TIMES = [12.0, 24.0, 36.0]


def cox_frame(frame: pd.DataFrame, score: str) -> CoxPHFitter:
    train = frame.loc[frame.group == "train", ["duration", "event", score]].copy()
    model = CoxPHFitter(penalizer=0.01)
    model.fit(train, duration_col="duration", event_col="event")
    return model


CLINICAL_VARS = ["Child_Pugh", "Gender", "Total_SLD_cm", "Age", "Diabetes", "AFP", "CA199", "Lesion_Count", "Hepatitis_C"]


def probability_model(frame: pd.DataFrame, score: str, clinical_input=None):
    """Use the native joint-clinical baseline; keep fusion score mappings fixed to Train."""
    if score != "Clinical-only_Score":
        return cox_frame(frame, score), frame, [score]
    if set(CLINICAL_VARS).issubset(frame.columns):
        source = frame
    else:
        if clinical_input is None:
            raise ValueError("Native joint-clinical probabilities require --clinical-input with the original nine coded variables")
        source = clinical_input.copy()
        if "ID" not in source or source.ID.astype(str).duplicated().any():
            raise ValueError("Clinical input requires unique patient ID values")
        source["ID"] = source.ID.astype(str)
        left = frame.copy()
        left["ID"] = left.ID.astype(str)
        source = left.merge(source[["ID"] + CLINICAL_VARS], on="ID", how="left", validate="one_to_one", sort=False)
    source = source.copy()
    source[CLINICAL_VARS] = source[CLINICAL_VARS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(source[CLINICAL_VARS].to_numpy(float)).all():
        raise ValueError("Clinical predictors are incomplete or non-finite; no implicit imputation is performed")
    train = source.loc[source.group == "train", ["duration", "event"] + CLINICAL_VARS]
    model = CoxPHFitter().fit(train, duration_col="duration", event_col="event")
    return model, source, CLINICAL_VARS


def km_survival_at(frame: pd.DataFrame, time: float) -> float:
    km = KaplanMeierFitter().fit(frame.duration, frame.event)
    return float(km.survival_function_at_times(time).iloc[0])


def censoring_survival_at(frame: pd.DataFrame, time: float) -> float:
    reverse = 1 - frame.event.astype(int)
    km = KaplanMeierFitter().fit(frame.duration, reverse)
    return float(km.survival_function_at_times(time).iloc[0])


def ph_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    from lifelines.statistics import proportional_hazard_test
    rows = []
    train = frame.loc[frame.group == "train"].copy()
    for score in SCORES:
        model = cox_frame(frame, score)
        values = proportional_hazard_test(model, train[["duration", "event", score]], time_transform=["rank", "km"]).summary.reset_index()
        for _, row in values.iterrows():
            rows.append({"model": score, "term": row.iloc[0], "time_transform": row.iloc[1], "chi_square": float(row["test_statistic"]), "p": float(row["p"])})
    return pd.DataFrame(rows)


def risk_censoring_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("train", "test1", "test2"):
        subset = frame.loc[frame.group == group]
        for time in TIMES:
            rows.append({"group": group, "time_months": time, "n": len(subset), "events_by_time": int(((subset.event == 1) & (subset.duration <= time)).sum()), "at_risk": int((subset.duration >= time).sum()), "censored_by_time": int(((subset.event == 0) & (subset.duration <= time)).sum()), "km_event_probability": 1 - km_survival_at(subset, time), "reverse_km_censor_survival": censoring_survival_at(subset, time)})
    return pd.DataFrame(rows)


def calibration(frame: pd.DataFrame, clinical_input=None) -> pd.DataFrame:
    rows = []
    for score in SCORES:
        model, prediction_frame, predictors = probability_model(frame, score, clinical_input)
        train = prediction_frame.loc[prediction_frame.group == "train"].copy()
        train["risk_group"] = pd.qcut(train[score], q=3, labels=["low", "middle", "high"], duplicates="drop")
        for group_name, group_frame in train.groupby("risk_group", observed=False):
            for time in TIMES:
                observed = 1 - km_survival_at(group_frame, time)
                predicted = float(1 - model.predict_survival_function(group_frame[predictors], times=[time]).mean(axis=1).mean())
                rows.append({"model": score, "risk_group": str(group_name), "time_months": time, "n": len(group_frame), "observed_event_probability_km": observed, "mean_predicted_event_probability": predicted, "calibration_error_observed_minus_predicted": observed - predicted})
    return pd.DataFrame(rows)


def brier(frame: pd.DataFrame, clinical_input=None) -> pd.DataFrame:
    rows = []
    train = frame.loc[frame.group == "train"].copy()
    test1 = frame.loc[frame.group == "test1"].copy()
    y_train = Surv.from_dataframe("event", "duration", train)
    y_test = Surv.from_dataframe("event", "duration", test1)
    eval_times = np.array(TIMES)
    for score in SCORES:
        model, prediction_frame, predictors = probability_model(frame, score, clinical_input)
        evaluation = prediction_frame.loc[prediction_frame.group == "test1"]
        predictions = model.predict_survival_function(evaluation[predictors], times=eval_times).T.to_numpy()
        try:
            ibs = float(integrated_brier_score(y_train, y_test, predictions, eval_times))
            status, error = "complete", ""
        except Exception as exc:
            ibs, status, error = np.nan, "failed", repr(exc)
        rows.append({"model": score, "group": "test1", "integrated_brier_score_12_24_36m": ibs, "status": status, "error": error})
    return pd.DataFrame(rows)


def dca_screen(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"status": "deferred", "reason": "A valid survival DCA requires horizon-specific IPCW event probabilities; no naive uncensored DCA was substituted."}])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_FILE)
    frame["event"] = frame.event.astype(int)
    frame["duration"] = frame.duration.astype(float)
    ph = ph_diagnostics(frame)
    rc = risk_censoring_diagnostics(frame)
    cal = calibration(frame, CLINICAL_INPUT)
    ibs = brier(frame, CLINICAL_INPUT)
    dca = dca_screen(frame)
    ph.to_csv(OUTPUT_DIR / "ph_diagnostics.csv", index=False, encoding="utf-8-sig")
    rc.to_csv(OUTPUT_DIR / "risk_set_censoring_diagnostics.csv", index=False, encoding="utf-8-sig")
    cal.to_csv(OUTPUT_DIR / "survival_calibration_tertiles.csv", index=False, encoding="utf-8-sig")
    ibs.to_csv(OUTPUT_DIR / "test1_integrated_brier_score.csv", index=False, encoding="utf-8-sig")
    dca.to_csv(OUTPUT_DIR / "dca_status.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "p1_diagnostics_manifest.json").write_text(json.dumps({"status": "complete_with_dca_deferred", "times_months": TIMES, "models": SCORES, "notes": ["PH evaluated on Train-fitted Cox models.", "Risk-set and censoring diagnostics are descriptive.", "Calibration uses Train tertiles and KM event probabilities.", "Brier/iBS is Test1 IPCW using Train censoring distribution.", "DCA was not replaced by a naive uncensored calculation."]}, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_cli():
    global INPUT_FILE, OUTPUT_DIR, CLINICAL_INPUT
    parser = argparse.ArgumentParser(description="Train-fitted single-score probabilities, numeric PH tests and IPCW IBS")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--clinical-input", type=Path, required=True, help="Original nine coded clinical variables and matching patient IDs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    INPUT_FILE, OUTPUT_DIR = args.scores.resolve(), args.output.resolve()
    CLINICAL_INPUT = pd.read_csv(args.clinical_input, dtype={"ID": str})


if __name__ == "__main__":
    configure_cli()
    main()
