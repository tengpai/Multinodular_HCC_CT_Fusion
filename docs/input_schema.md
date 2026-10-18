# Private input schemas

These are interfaces, not public patient tables. Preserve the study's original
clinical coding and feature transformations. Do not silently fill missing values
or recompute means/scales using validation patients.

## Frozen inference

Required component columns: `Clinical_Score`, `Rad_Diff_HighRisk`,
`DL_Heterogeneity`, `DL_MaxPooling`, `Habitat_Score`. `ID` is optional; if present
it must be unique. All required scores must be finite numbers. They are the
**unstandardized component scores** before fusion-stage standardization.

- `Clinical_Score`: nine-variable weighted score. Separate training-set
  univariable Cox models (L2 0.1) estimated weights on the original-coded inputs;
  these weights were applied to training-standardized variables.
- `Rad_Diff_HighRisk`: negative model-predicted expected survival time from the
  12-feature Rad_Diff Cox model (L2 0.1), not its linear predictor. Higher is riskier.
- `DL_Heterogeneity`, `DL_MaxPooling`: the original locked component scores.
  Use their original sign and scaling; do not optimize sign on test outcomes.
- `Habitat_Score`: log partial hazard from the locked max-H2 model.

`--compute-clinical` requires `Child_Pugh`, `Gender`, `Total_SLD_cm`, `Age`,
`Diabetes`, `AFP`, `CA199`, `Lesion_Count`, `Hepatitis_C`.
Age is years, total lesion diameter is cm, and lesion count is a count.
The original coding is Child-Pugh A/B = 0/1, Gender female/male = 0/1,
and Diabetes and hepatitis C absent/present = 0/1. AFP and CA199 are the
source table's 0/1 indicators using thresholds of 400 ng/mL and 37 U/mL,
respectively. This interface does not accept raw AFP/CA199 values. Preserve
the source table's threshold-boundary convention when constructing new inputs;
do not infer it from binary data or rounded manuscript labels.

Output: `fusion_score`, `risk_group` (High when score >= training median),
`training_median_cutoff`, and optional `ID`. It is a relative risk score, not a
survival probability or a clinical treatment recommendation.

## Locked fusion reconstruction and evaluation

The reconstruction input contains unique `ID`, `group` (`train`, `test1`,
`test2`), `duration` in months, `event` (0 censored, 1 event), the nine clinical
variables and three non-habitat imaging scores above. Exact study counts are
282/113/491. A private habitat directory must contain the original complete
hash-verified artifact set (`model_lock_manifest.json`, model and candidate
pickles, and all listed artifacts). Only load trusted study pickles.

`fusion_scores_locked.csv`, produced by reconstruction, is the common input to
statistics and diagnostics. It includes the component scores and
`Clinical-only_Score`, `Clinical_RadDiff_DLHet_Score`,
`Four-score_without_Habitat_Score`, `Full_five-score_Fusion_Score`.

For `clinical_benchmarks`, use the private merged evaluation table with those
four model columns and `CNLC_rank`, `BCLC_rank`, `Up_to_7_rank`, `ALBI_rank`.
Ordinal ranks must increase with clinical risk: CNLC Ib/IIa/IIb, BCLC A/B,
Up-to-7 within/beyond, and ALBI grades 1/2/3. Rank offsets do not affect C-index,
but directions must not be selected from observed outcomes.

RFS cohorts use surgery as time zero and recurrence/death as the event. Test set
2 uses PFS from first treatment to progression/death. Analyze these endpoints separately, using the outcome definitions recorded in
the study data.

## Habitat source matrices

Use the original private wide habitat matrix and aligned labels, not a
preselected feature subset. Candidate parsing and ID normalization are defined
by `habitat_train_only.py`. There are 12 combinations (mean/min/max ×
H1/H2/H3/Merged). Default input SHA-256 checks protect exact reproduction;
overriding a hash explicitly defines a new run rather than the published run.
