# Changelog

## v2.0.1 (2026-10-23)

Documentation and notebook clean-up; no change to analysis code, model parameters or reported results.

- Score-construction notebooks: risk-group plots now use the training-set median cutoff, as in the manuscript. An exploratory log-rank cutoff search that was not used for any reported result was removed.
- Removed an unused PCA setting from the deep-learning configuration files.
- Revised the README and the clinical score specification for clarity.

## v2.0.0 (2026-10-20)

- Main Clinical_Score changed to the training-fitted nine-variable multivariable Cox linear predictor; added the post hoc sensitivity parameter file and clinical-plus-fusion validation utilities. Imaging component scores and the max-H2 habitat model unchanged.

## v1.0.0 (2026-10-18)

- Initial public release of the max-H2 habitat selection, locked score fusion and evaluation code.
