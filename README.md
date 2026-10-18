# Multinodular HCC CT fusion analysis

Core code for **CT-based Tumor Heterogeneity Fusion Model for Prognosis in
Multinodular Hepatocellular Carcinoma: A Multicenter Study**.

This repository contains the current max-H2 habitat selection, locked score
fusion, survival evaluation and clinical benchmark comparisons. It also includes
component correlation/VIF, model contribution, calibration, proportional-hazards
diagnostics and fusion-stage bootstrap validation. Scientific analysis functions
are retained from the study code; this distribution reorganizes their public scope.

## Installation

Use Python 3.12 in an isolated environment. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Usage

Apply the fixed fusion parameters to an appropriately constructed component-score
table without fitting on the supplied patients:

```sh
python -m current_analysis.locked_inference --input /path/component_scores.csv --output /path/predictions.csv
```

`config/locked_fusion_model.json` contains aggregate model parameters and training
cutoffs. It contains no patient records. See [input definitions](docs/input_schema.md)
and [analysis commands](docs/reproducibility.md) for other entry points.

## Contents

- `01_score_construction/`: original feature-extraction notebooks and configurations
  for inter-nodule radiomics and fixed ImageNet convolutional-network features.
- `current_analysis/`: current max-H2 selection, score fusion and evaluation.
- `config/`: aggregate locked parameters and reference extraction settings.
- `docs/`: input definitions and commands.
- `tests/`: synthetic-data checks.

The notebooks require the original OneKey (`onekey_algo`) environment and the
relevant imaging libraries; installing the statistical requirements does not
provide that environment. Notebook outputs are cleared. Raw CT preprocessing,
manual segmentation and WES processing are not reproduced by the statistical
entry points alone.

## Data and access

Raw CT, WES and patient-level tables are not included. Deidentified feature,
component-score and necessary clinical tables may be requested from the
corresponding authors, subject to institutional approval and a data-use agreement.
Exact study reproduction requires the approved inputs and locked artifacts.

The surgical cohorts evaluate recurrence-free survival. The nonsurgical cohort
evaluates exploratory progression-free-survival transfer and is analyzed
separately. Bootstrap validation of the fusion stage fixes the component scores;
it does not repeat their upstream feature selection.

## Citation and license

Use the repository URL and the exact commit used for an analysis. `CITATION.cff`
provides software citation metadata. The original research-use license is retained
in `LICENSE.txt`; third-party software remains under its own licenses.
