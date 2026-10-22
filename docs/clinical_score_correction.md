# Clinical score specification

The main Clinical_Score is the centered linear predictor of a nine-variable multivariable Cox model fitted in the training set using the original variable coding. The frozen mean vector and coefficients are stored in `config/locked_fusion_model.json`; the scale vector is one. The fusion stage uses training-set standardization and an L2 penalty of 0.01.

The sensitivity specification fits each training-standardized clinical variable separately (L2 = 0.1) and sums the coefficient-times-standardized-variable contributions. Its fixed parameters are stored in `config/sensitivity_fusion_model.json`.

No coefficient or cutoff was estimated in the external cohorts. As described in Appendix S1.1 of the manuscript, this specification was adopted on methodological grounds after the initial external evaluation, and results for both specifications are reported. The four imaging component scores are unchanged.

`current_analysis/clinical_fusion_validation.py` provides the clinical-plus-fusion bootstrap and the cohort-specific IPCW AUC convention. Input tables, images, sequencing data and patient-level scores are not distributed.
