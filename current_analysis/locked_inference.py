"""Apply aggregate locked parameters without fitting to new patients."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

DEFAULT_PARAMETERS = Path(__file__).resolve().parents[1] / "config" / "locked_fusion_model.json"


def load_parameters(path=DEFAULT_PARAMETERS):
    with Path(path).open(encoding="utf-8") as handle:
        parameters = json.load(handle)
    if parameters.get("habitat_candidate") != "max-H2":
        raise ValueError("This release requires the locked max-H2 habitat model")
    return parameters


def numeric_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate input columns")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Missing or non-finite values; do not silently impute during inference")
    return values


def clinical_score(frame, parameters):
    """Weights fitted on original-coded variables, applied to Train-standardized values."""
    spec = parameters["clinical_score"]
    columns = spec["clinical_variables"]
    values = numeric_columns(frame, columns)
    mean = np.asarray(spec["scaler_mean"], dtype=float)
    scale = np.asarray(spec["scaler_scale"], dtype=float)
    beta = np.asarray([spec["coefficients"][c] for c in columns], dtype=float)
    if len(columns) != len(mean) or len(mean) != len(scale) or (scale <= 0).any():
        raise ValueError("Invalid clinical standardization parameters")
    return pd.Series(((values.to_numpy(float) - mean) / scale) @ beta,
                     index=frame.index, name="Clinical_Score")


def predict_fusion(frame, parameters, model_name="Full_five-score_Fusion"):
    spec = parameters["fusion_models"][model_name]
    columns = spec["predictors"]
    values = numeric_columns(frame, columns)
    means = np.asarray([spec["scalers"][p]["mean"] for p in columns], dtype=float)
    scales = np.asarray([spec["scalers"][p]["scale"] for p in columns], dtype=float)
    beta = np.asarray([spec["coefficients"][p + "_Z"] for p in columns], dtype=float)
    if (scales <= 0).any() or not np.isfinite(np.r_[means, scales, beta]).all():
        raise ValueError("Invalid frozen model parameters")
    scores = ((values.to_numpy(float) - means) / scales) @ beta
    return pd.DataFrame({
        "fusion_score": scores,
        "risk_group": np.where(scores >= spec["cutoff_train_median"], "High", "Low"),
        "training_median_cutoff": spec["cutoff_train_median"],
    }, index=frame.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--compute-clinical", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the input")
    frame = pd.read_csv(args.input)
    parameters = load_parameters(args.parameters)
    if args.compute_clinical:
        frame["Clinical_Score"] = clinical_score(frame, parameters)
    prediction = predict_fusion(frame, parameters)
    if "ID" in frame:
        if frame["ID"].duplicated().any():
            raise ValueError("Duplicate patient identifiers")
        prediction.insert(0, "ID", frame["ID"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f"Applied locked max-H2 fusion parameters to {len(frame)} records; no fitting performed.")


if __name__ == "__main__":
    main()
