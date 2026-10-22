# Clinical-score correction

The main Clinical_Score is a training-only nine-variable joint Cox centered linear predictor in original variable coding. Its frozen mean vector and coefficients are in `config/locked_fusion_model.json`; the scale vector is one. The fusion stage retains training standardization and L2=0.01.

The post hoc sensitivity specification fits each training-standardized clinical variable separately (L2=0.1), then sums the resulting coefficient-times-standardized-variable contributions. Use `config/sensitivity_fusion_model.json` for fixed-parameter sensitivity inference.

No coefficient or cutoff is chosen from external outcomes. The correction follows prior examination of those cohorts and must not be described as newly untouched validation. The four imaging component scores remain fixed.

`clinical_fusion_validation.py` provides the clinical-plus-fusion bootstrap and cohort-specific IPCW AUC convention. Record numerical failures and sparse-category warnings. The original input tables, images, WES, detailed audit logs and patient scores are not distributed.
