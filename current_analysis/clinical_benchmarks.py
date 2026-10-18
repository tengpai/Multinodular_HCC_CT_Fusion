"""Within-cohort clinical benchmarking using one paired patient-bootstrap stream."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

MODEL_COLUMNS = {
    "Clinical Score": "Clinical-only_Score",
    "Three-score model": "Clinical_RadDiff_DLHet_Score",
    "Four-score model": "Four-score_without_Habitat_Score",
    "Five-score model": "Full_five-score_Fusion_Score",
    "CNLC stage": "CNLC_rank", "BCLC stage": "BCLC_rank",
    "Up-to-7": "Up_to_7_rank", "ALBI grade": "ALBI_rank",
}
STAGING = ["CNLC stage", "BCLC stage", "Up-to-7", "ALBI grade"]


def pair_matrix(times, events):
    return ((events[:, None] == 1) &
            ((times[:, None] < times[None, :]) |
             ((times[:, None] == times[None, :]) & (events[None, :] == 0)))).astype(float)


def bootstrap_scores(times, events, scores, rng, resamples=2000):
    times, raw_events = np.asarray(times, float), np.asarray(events)
    if not np.isin(raw_events, [0, 1]).all():
        raise ValueError("Events must be binary before conversion")
    events = raw_events.astype(int)
    if len(times) != len(events) or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("Invalid survival times")
    if not np.isin(events, [0, 1]).all() or resamples < 1:
        raise ValueError("Invalid events or resample count")
    n = len(times)
    permitted = pair_matrix(times, events)
    if permitted.sum() == 0:
        raise ValueError("No comparable survival pairs")
    indices = rng.integers(0, n, size=(resamples, n))
    counts = np.array([np.bincount(row, minlength=n) for row in indices], dtype=float)
    denominator = np.einsum("bi,bi->b", counts @ permitted, counts)
    valid = denominator > 0
    if not valid.all():
        raise ValueError("Bootstrap contains a replicate with no comparable pairs")
    points, boots = {}, {}
    for name, risk in scores.items():
        risk = np.asarray(risk, float)
        if len(risk) != n or not np.isfinite(risk).all():
            raise ValueError(f"Invalid risk values for {name}")
        credit = permitted * ((risk[:, None] > risk[None, :]).astype(float) +
                              .5 * (risk[:, None] == risk[None, :]))
        point = float(credit.sum() / permitted.sum())
        if not np.isclose(point, concordance_index(times, -risk, events), atol=1e-12, rtol=0):
            raise RuntimeError("Independent pair-count check disagrees with lifelines")
        values = np.einsum("bi,bi->b", counts @ credit, counts) / denominator
        for b in [i for i in (0, 19, 137, 999, 1999) if i < resamples]:
            sample = np.repeat(np.arange(n), counts[b].astype(int))
            expected = concordance_index(times[sample], -risk[sample], events[sample])
            if not np.isclose(values[b], expected, atol=1e-12, rtol=0):
                raise RuntimeError("Bootstrap library verification failed")
        points[name], boots[name] = point, values
    return points, boots


def evaluate(frame, resamples=2000, seed=20260718):
    required = {"group", "duration", "event", *MODEL_COLUMNS.values()}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if set(frame["group"]) != {"train", "test1", "test2"}:
        raise ValueError("Expected the three prespecified cohorts")
    if "ID" in frame and frame["ID"].duplicated().any():
        raise ValueError("Patient IDs must be unique across cohorts")
    rng = np.random.default_rng(seed)
    results, differences = [], []
    for group in ("train", "test1", "test2"):
        d = frame.loc[frame["group"] == group]
        points, boot = bootstrap_scores(d.duration.to_numpy(float), d.event.to_numpy(),
            {name: d[col].to_numpy(float) for name, col in MODEL_COLUMNS.items()}, rng, resamples)
        endpoint = "PFS" if group == "test2" else "RFS"
        for name, point in points.items():
            low, high = np.quantile(boot[name], [.025, .975])
            results.append({"cohort": group, "endpoint": endpoint, "n": len(d),
                "events": int(d.event.sum()), "predictor": name, "c_index": point,
                "ci_lower": float(low), "ci_upper": float(high),
                "successful_bootstrap": resamples})
        for model in list(MODEL_COLUMNS)[:4]:
            for comparator in STAGING:
                low, high = np.quantile(boot[model] - boot[comparator], [.025, .975])
                differences.append({"cohort": group, "endpoint": endpoint, "n": len(d),
                    "model": model, "comparator": comparator,
                    "delta_c_index": points[model] - points[comparator],
                    "ci_lower": float(low), "ci_upper": float(high),
                    "successful_paired_bootstrap": resamples,
                    "comparison_role": {"train": "apparent training performance",
                        "test1": "primary external evaluation", "test2": "exploratory PFS transfer"}[group]})
    return pd.DataFrame(results), pd.DataFrame(differences)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Restricted evaluation table; see docs/input_schema.md")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=2000)
    args = p.parse_args()
    points, differences = evaluate(pd.read_csv(args.input), args.resamples)
    args.output.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output / "cohort_cindex.csv", index=False)
    differences.to_csv(args.output / "paired_cindex_differences.csv", index=False)
    (args.output / "method.json").write_text(json.dumps({
        "seed": 20260718, "resamples": args.resamples, "refitting": False,
        "paired": True, "random_stream": "sequential train/test1/test2",
        "test2_interpretation": "exploratory transfer; do not pool with RFS cohorts",
        "intervals": "unadjusted percentile intervals; no blanket superiority claim",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
