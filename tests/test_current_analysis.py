import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from current_analysis.locked_inference import load_parameters, clinical_score, predict_fusion
from current_analysis.clinical_benchmarks import pair_matrix, bootstrap_scores
from current_analysis import habitat_train_only as habitat
from current_analysis import fusion_locked_core as fusion


class LockedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_parameters()
        self.model = self.parameters["fusion_models"]["Full_five-score_Fusion"]
        self.mean_scores = pd.DataFrame([{p: self.model["scalers"][p]["mean"]
            for p in self.model["predictors"]}])

    def test_mean_scores_have_zero_linear_predictor(self):
        self.assertAlmostEqual(predict_fusion(self.mean_scores, self.parameters).fusion_score.iloc[0], 0.)

    def test_predictions_do_not_depend_on_external_batch(self):
        one = predict_fusion(self.mean_scores, self.parameters).fusion_score.iloc[0]
        other = self.mean_scores * 200 + 100
        batch = pd.concat([self.mean_scores, other], ignore_index=True)
        self.assertAlmostEqual(predict_fusion(batch, self.parameters).fusion_score.iloc[0], one)

    def test_score_increases_by_the_frozen_coefficient(self):
        frame = self.mean_scores.copy()
        p = self.model["predictors"][0]
        frame[p] += self.model["scalers"][p]["scale"]
        expected = self.model["coefficients"][p + "_Z"]
        self.assertAlmostEqual(predict_fusion(frame, self.parameters).fusion_score.iloc[0], expected)

    def test_missing_component_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_fusion(self.mean_scores.drop(columns=["Habitat_Score"]), self.parameters)

    def test_nonfinite_input_is_not_silently_imputed(self):
        frame = self.mean_scores.copy()
        frame.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            predict_fusion(frame, self.parameters)

    def test_clinical_score_uses_locked_original_scale_weights(self):
        spec = self.parameters["clinical_score"]
        frame = pd.DataFrame([spec["scaler_mean"]], columns=spec["clinical_variables"])
        self.assertAlmostEqual(clinical_score(frame, self.parameters).iloc[0], 0.)
        first = spec["clinical_variables"][0]
        frame[first] += spec["scaler_scale"][0]
        self.assertAlmostEqual(clinical_score(frame, self.parameters).iloc[0], spec["coefficients"][first])

    def test_invalid_habitat_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.json"
            p.write_text(json.dumps({"habitat_candidate": "max-H1"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_parameters(p)


class SurvivalComparisonTests(unittest.TestCase):
    def test_event_censoring_ties_match_harrell_definition(self):
        time = np.array([1., 1., 2., 3.])
        event = np.array([1, 0, 1, 0])
        risk = np.array([4., 3., 2., 1.])
        permitted = pair_matrix(time, event)
        self.assertEqual(permitted[0, 1], 1)
        self.assertEqual(permitted[1, 0], 0)
        credit = permitted * ((risk[:, None] > risk[None, :]) +
                              .5 * (risk[:, None] == risk[None, :]))
        self.assertAlmostEqual(credit.sum() / permitted.sum(), concordance_index(time, -risk, event))

    def test_paired_identical_predictors_have_identical_bootstrap(self):
        n = 40
        t = np.arange(1., n + 1)
        e = np.ones(n, dtype=int)
        risk = -t
        p, b = bootstrap_scores(t, e, {"one": risk, "same": risk}, np.random.default_rng(7), 50)
        self.assertEqual(p["one"], 1.)
        np.testing.assert_array_equal(b["one"], b["same"])

    def test_nonbinary_events_are_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_scores(np.arange(1., 11.), np.r_[.5, np.ones(9)],
                             {"risk": np.arange(10.)}, np.random.default_rng(1), 5)

    def test_no_comparable_pairs_is_reported(self):
        with self.assertRaises(ValueError):
            bootstrap_scores(np.ones(8), np.zeros(8), {"risk": np.arange(8)},
                             np.random.default_rng(7), 10)


class HabitatGateTests(unittest.TestCase):
    def test_folds_are_reproducible_and_exhaustive(self):
        labels = pd.DataFrame({"event": [0, 1] * 20})
        a = habitat.fixed_folds(labels, habitat.SEED)
        np.testing.assert_array_equal(a, habitat.fixed_folds(labels, habitat.SEED))
        self.assertEqual(set(a), set(range(5)))
        self.assertTrue((np.bincount(a) == 8).all())

    def test_validation_uses_training_preprocessing(self):
        state = {"available_columns": ["a"], "medians": {"a": 10.},
                 "means": {"a": 10.}, "scales": {"a": 2.}, "univariate_columns": ["a"]}
        frame = pd.DataFrame({"a": [14., np.nan, 1000.]})
        out = habitat.transform_preprocessor(frame, state)
        np.testing.assert_allclose(out["a"], [2., 0., 495.])

    def test_modified_artifact_blocks_external_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "final_habitat_model.pkl"
            artifact.write_bytes(b"test artifact; never unpickled")
            candidate = root / "candidate_max-H2.pkl"
            candidate.write_bytes(b"test candidate")
            manifest = root / "model_lock_manifest.json"
            record = {"path": artifact.name, "bytes": artifact.stat().st_size,
                      "sha256": fusion.sha256(artifact)}
            manifest.write_text(json.dumps({"status": "locked", "candidate": "max-H2",
                                            "artifacts": [record]}), encoding="utf-8")
            with patch.multiple(fusion, HABITAT_DIR=root, HABITAT_MANIFEST=manifest,
                                HABITAT_MODEL=artifact, HABITAT_CANDIDATE=candidate):
                self.assertEqual(fusion.verify_habitat_lock()["candidate"], "max-H2")
                artifact.write_bytes(b"modified")
                with self.assertRaises(RuntimeError):
                    fusion.verify_habitat_lock()


if __name__ == "__main__":
    unittest.main()
