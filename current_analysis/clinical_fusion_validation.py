"""Clinical/fusion-stage bootstrap and original cohort-censor AUC convention."""
import warnings
import numpy as np
# Compatibility with the pinned scikit-survival release on NumPy 2.4+.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
import pandas as pd
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from .fusion_locked_core import build_clinical_scores, fit_model

PREDICTORS = ["Clinical_Score", "Rad_Diff_HighRisk", "DL_Heterogeneity", "DL_MaxPooling", "Habitat_Score"]

def cohort_ipcw_auc(frame, score="Full_five-score_Fusion_Score", times=(12., 24., 36.)):
    """Estimate censoring within the evaluation cohort, matching the manuscript."""
    y = Surv.from_arrays(frame.event.astype(bool), frame.duration.astype(float))
    return cumulative_dynamic_auc(y, y, frame[score].to_numpy(float), np.asarray(times))[0]

def bootstrap_clinical_fusion(train, n_resamples=1000, seed=20261090, method="joint"):
    """Refit clinical and fusion stages; freeze imaging scores and clinical list.

    Records warnings and failures rather than silently retaining only stable fits.
    This does not validate the upstream imaging feature selection or imputation.
    """
    train = train.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    records = []
    for b in range(n_resamples):
        sampled = train.iloc[rng.integers(len(train), size=len(train))].copy()
        sampled["group"] = "train"
        evaluation = train.copy()
        evaluation["group"] = "evaluation"
        joined = pd.concat([sampled, evaluation], ignore_index=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            def fit_once(options=None):
                scored, meta = build_clinical_scores(joined, method, fit_options=options)
                _, scored = fit_model(scored, "Full_five-score_Fusion", PREDICTORS, fit_options=options)
                values = []
                for group in ("train", "evaluation"):
                    a = scored[scored.group == group]
                    values.append(concordance_index(a.duration, -a["Full_five-score_Fusion_Score"], a.event))
                return values, meta
            retried = False
            try:
                try:
                    values, clinical_meta = fit_once()
                    if any("failed to converge" in str(w.message).lower() for w in caught):
                        retried = True
                        values, clinical_meta = fit_once({"step_size": 0.2, "max_steps": 1000})
                except ConvergenceError:
                    retried = True
                    values, clinical_meta = fit_once({"step_size": 0.2, "max_steps": 1000})
                rec = {"replicate": b, "apparent": values[0], "original_sample": values[1],
                       "optimism": values[0] - values[1], "error": "",
                       "constant_variables": clinical_meta["constant_variables"]}
            except Exception as error:
                rec = {"replicate": b, "error": type(error).__name__ + ": " + str(error)}
            rec["numerical_retry_same_sample_and_model"] = retried
            rec["warnings"] = sorted(set(str(w.message) for w in caught
                                         if not issubclass(w.category, DeprecationWarning)))
        records.append(rec)
    return records
