# Analysis commands

Use Python 3.12 and run from the repository root. All paths are placeholders for
approved study inputs and output locations. No patient data are bundled.

## Habitat selection

```sh
python -m current_analysis.habitat_train_only --raw /path/habitat_raw.csv --labels /path/labels.csv --output /path/habitat_run --stage prepare
python -m current_analysis.habitat_train_only --raw /path/habitat_raw.csv --labels /path/labels.csv --output /path/habitat_run --stage select
```

Use the same inputs and code for both stages. Preserve the generated model lock
manifest and artifacts. The selected study component is max-H2. Candidate
comparison and nested candidate-selection assessment are separate outputs.

## Fusion reconstruction and evaluation

```sh
python -m current_analysis.fusion_locked_core --habitat-dir /path/habitat_run --fusion-input /path/fusion_input.csv --output /path/fusion_run
python -m current_analysis.fusion_statistics --scores /path/fusion_run/fusion_scores_locked.csv --output /path/statistics
python -m current_analysis.clinical_benchmarks --input /path/evaluation_input.csv --output /path/benchmarks
python -m current_analysis.complementarity --scores /path/fusion_run/fusion_scores_locked.csv --manifest /path/fusion_run/fusion_core_manifest.json --output /path/complementarity
python -m current_analysis.probability_diagnostics --scores /path/fusion_run/fusion_scores_locked.csv --output /path/probability
python -m current_analysis.fusion_diagnostics --scores /path/fusion_run/fusion_scores_locked.csv --manifest /path/fusion_run/fusion_core_manifest.json --output /path/diagnostics --bootstrap 1000
```

Run `python -m MODULE --help` for arguments. Input definitions are in
`input_schema.md`. Fusion scaling and Cox coefficients are estimated using the
training set only. Validation patients do not alter them. Clinical benchmarks use
the same patients and endpoint within each cohort. Model contribution and
permutation importance describe the fixed model and do not refit a reduced model.
Calibration probabilities use a training-fitted Cox model with each locked score
as its sole predictor. The fusion-stage bootstrap keeps component scores fixed.

The feature-extraction notebooks require the original OneKey environment.
The statistical commands require the approved feature/score tables and artifacts;
they do not process raw CT or WES. Keep generated patient-level outputs outside
the repository.
