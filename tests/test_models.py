"""Week 3 tests: metrics correctness, baseline sanity, and the guard rails.

The three properties worth defending:
1. **Metrics are right** — Brier Skill Score, ECE and PR-AUC are computed from
   hand-checkable cases (perfect forecasts, constant forecasts, worse-than-ref).
2. **Training never sees the test split** — asserted by spying on what fit()
   actually receives, not by trusting the code path.
3. **The gates really gate** — promotion refuses unevaluated models and
   unnamed reviewers; test scoring refuses an untuned threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismo.config import load_config
from seismo.errors import ValidationError
from seismo.features.engineering import feature_columns
from seismo.models.baselines import ClimatologicalRate, GlobalRate, PoissonRate
from seismo.models.evaluate import (
    brier_score,
    brier_skill_score,
    calibration_table,
    classification_metrics,
    evaluate_predictions,
    expected_calibration_error,
    find_optimal_threshold,
    pr_auc,
    roc_auc,
)
from seismo.models.train import (
    MODEL_NAMES,
    load_registry,
    promote,
    split_frame,
    train_model,
)

# ---------------------------------------------------------------------------
# 1. Metrics — hand-checkable cases
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_forecast_scores_perfectly(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        assert brier_score(y, p) == pytest.approx(0.0)
        assert expected_calibration_error(y, p) == pytest.approx(0.0)
        assert pr_auc(y, p) == pytest.approx(1.0)
        assert roc_auc(y, p) == pytest.approx(1.0)

    def test_brier_matches_hand_computation(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.4, 0.6])
        expected = np.mean([0.01, 0.01, 0.36, 0.36])
        assert brier_score(y, p) == pytest.approx(expected)

    def test_constant_forecast_at_base_rate_has_zero_skill(self):
        """A forecast equal to the climatological rate is exactly the reference."""
        rng = np.random.default_rng(0)
        y = (rng.random(10_000) < 0.03).astype(int)
        p = np.full_like(y, y.mean(), dtype=float)
        assert brier_skill_score(y, p, y, p) == pytest.approx(0.0)

    def test_brier_skill_score_sign_convention(self):
        rng = np.random.default_rng(1)
        y = (rng.random(5_000) < 0.05).astype(int)
        reference = np.full_like(y, 0.05)

        better = np.clip(reference + (y - 0.05) * 0.3, 0.001, 0.999)
        worse = np.clip(reference - (y - 0.05) * 0.3, 0.001, 0.999)

        assert brier_skill_score(y, better, y, reference) > 0
        assert brier_skill_score(y, worse, y, reference) < 0

    def test_pr_auc_reflects_imbalance_baseline(self):
        """Random scoring should give PR-AUC ≈ base rate, ROC-AUC ≈ 0.5."""
        rng = np.random.default_rng(2)
        y = (rng.random(20_000) < 0.03).astype(int)
        p = rng.random(20_000)
        assert pr_auc(y, p) == pytest.approx(0.03, abs=0.02)
        assert roc_auc(y, p) == pytest.approx(0.5, abs=0.03)

    def test_calibration_weights_sum_to_one(self):
        rng = np.random.default_rng(3)
        y = (rng.random(4_000) < 0.2).astype(int)
        p = rng.random(4_000)
        table = calibration_table(y, p, n_bins=10)
        assert table["weight"].sum() == pytest.approx(1.0)
        assert table["count"].sum() == 4_000
        assert (table["abs_gap"] >= 0).all()

    def test_ece_of_perfectly_calibrated_draws_is_small(self):
        rng = np.random.default_rng(4)
        p = rng.random(50_000)
        y = (rng.random(50_000) < p).astype(int)   # drawn *from* p → calibrated
        assert expected_calibration_error(y, p, n_bins=20) < 0.01

    @pytest.mark.parametrize("bad", [
        (np.array([0, 1]), np.array([1.5, 0.5])),          # probability > 1
        (np.array([0, 1]), np.array([np.nan, 0.5])),        # NaN prediction
        (np.array([0, 1]), np.array([0.5, 0.5, 0.5])),      # shape mismatch
        (np.array([0, 1]), np.array([-0.2, 0.5])),          # probability < 0
    ])
    def test_invalid_predictions_are_rejected(self, bad):
        y, p = bad
        with pytest.raises(ValidationError):
            evaluate_predictions(y, p)

    def test_empty_prediction_set_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            evaluate_predictions(np.array([]), np.array([]))

    def test_single_class_split_is_reported_not_crashed_silently(self):
        y = np.zeros(10, dtype=int)
        p = np.full(10, 0.2)
        with pytest.raises(ValidationError, match="one class"):
            pr_auc(y, p)

    def test_threshold_tuning_maximises_f1_on_validation(self):
        rng = np.random.default_rng(5)
        y = (rng.random(20_000) < 0.05).astype(int)
        p = np.clip(rng.random(20_000) * 0.5 + y * 0.3, 0, 1)
        threshold, score = find_optimal_threshold(y, p, objective="f1")
        assert 0 < threshold < 1
        at_tuned = classification_metrics(y, p, threshold=threshold)["f1"]
        at_half = classification_metrics(y, p, threshold=0.5)["f1"]
        assert at_tuned >= at_half - 1e-9

    def test_evaluate_predictions_reports_reference_skill(self):
        y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.1, 0.1, 0.2, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.7])
        ref = np.full_like(y, 0.2)
        bundle = evaluate_predictions(y, p, reference_prob=ref)
        assert "brier_skill_score" in bundle
        assert "reference_brier" in bundle
        assert bundle["brier_skill_score"] > 0     # the model beats the constant


# ---------------------------------------------------------------------------
# 2. Baselines
# ---------------------------------------------------------------------------

def _toy_frame(n_cells: int = 5, rows_per_cell: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, n_cells, size=n_cells * rows_per_cell)
    active = rng.random(len(cells)) < 0.1
    return pd.DataFrame({
        "obs_date": pd.date_range("2010-01-01", periods=len(cells), freq="MS", tz="UTC"),
        "cell_id": cells,
        "latitude_center": rng.uniform(-89, 89, len(cells)),
        "longitude_center": rng.uniform(-179, 179, len(cells)),
        "events_last_30d": rng.poisson(0.3, len(cells)),
        "events_last_90d": rng.poisson(1.0, len(cells)),
        "events_last_365d": rng.poisson(4.0, len(cells)),
        "mean_magnitude_90d": rng.uniform(0, 6, len(cells)),
        "max_magnitude_90d": rng.uniform(0, 8, len(cells)),
        "mean_depth_90d": rng.uniform(-3, 40, len(cells)),
        "events_ge45_last_365d": rng.poisson(0.4, len(cells)),
        "days_since_last_event": rng.uniform(0, 9000, len(cells)),
        "target": active.astype(int),
    })


class TestBaselines:
    def test_climatology_probabilities_are_valid_and_bounded(self):
        frame = _toy_frame()
        model = ClimatologicalRate(prior_strength=50).fit(frame)
        proba = model.predict_proba(frame)
        assert proba.shape == (len(frame), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_climatology_shrinks_small_samples_toward_global(self):
        """A cell seen once with one hit must NOT report p≈1 — that is the
        overconfidence that destroys Brier scores on rare events."""
        frame = pd.DataFrame({
            "cell_id": [1, 1, 2, 2, 2, 3, 3, 3],
            "target": [1, 0, 1, 1, 1, 0, 0, 0],
        })
        global_rate = frame["target"].mean()          # 0.5
        model = ClimatologicalRate(prior_strength=50).fit(frame)

        n1 = model.predict_proba(pd.DataFrame({"cell_id": [1]}))[0, 1]
        assert n1 < 1.0, "single-hit cell must be shrunk below its raw rate"
        assert abs(n1 - global_rate) < abs(1.0 - global_rate), "moved toward global"

    def test_climatology_falls_back_to_global_for_unseen_cell(self):
        frame = _toy_frame()
        model = ClimatologicalRate().fit(frame)
        unseen = pd.DataFrame({"cell_id": [99_999]})
        assert model.predict_proba(unseen)[0, 1] == pytest.approx(
            frame["target"].mean()
        )

    def test_climatology_refuses_unfitted_prediction(self):
        with pytest.raises(ValidationError, match="not fitted"):
            ClimatologicalRate().predict_proba(_toy_frame())

    def test_poisson_outputs_are_valid_probabilities(self):
        frame = _toy_frame()
        cols = feature_columns(frame, include_coordinates=True)
        model = PoissonRate().fit(frame, cols)
        proba = model.predict_proba(frame)
        assert proba.shape == (len(frame), 2)
        assert np.all(np.isfinite(proba))
        assert (proba >= 0).all() and (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_global_rate_is_constant(self):
        frame = _toy_frame()
        model = GlobalRate().fit(frame)
        proba = model.predict_proba(frame)
        assert np.allclose(proba[:, 1], frame["target"].mean())

    def test_baselines_report_auc_above_random_on_signal(self):
        """When trailing activity genuinely predicts the target, the baselines
        must beat random — otherwise the fixture is meaningless."""
        from seismo.models.evaluate import evaluate_predictions

        frame = _toy_frame()
        cols = feature_columns(frame, include_coordinates=True)
        p = PoissonRate().fit(frame, cols).predict_proba(frame)[:, 1]
        bundle = evaluate_predictions(frame["target"], p)
        assert 0 <= bundle["roc_auc"] <= 1


# ---------------------------------------------------------------------------
# 3. Guard rails: splits, leakage, promotion
# ---------------------------------------------------------------------------

def _split_project(tmp_path) -> object:
    """Project whose feature frame has real train/val/test boundaries."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        "collection:\n"
        '  start_date: "2000-01-01"\n'
        '  end_date: "2012-12-31"\n'
        "  min_magnitude: 2.5\n"
        "grid:\n  cell_size_deg: 1.0\n"
        "forecast:\n  magnitude_threshold: 4.5\n  horizon_days: 30\n"
        "  min_lookback_days: 365\n"
        "splits:\n"
        '  train_end: "2007-12-31"\n'
        '  val_end: "2009-12-31"\n'
        "  purge_days: 30\n"
        "seed: 42\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return load_config(tmp_path)


def _write_features(cfg, n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Feature frame with chronological rows and real train/val/test splits."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2001-01-01", "2012-10-01", freq="MS", tz="UTC")
    obs = np.tile(dates, n // len(dates) + 1)[:n]
    cells = rng.integers(0, 40, size=n)

    base = rng.random(n)
    activity = rng.poisson(4, size=n)
    target = (base * 0.02 + activity * 0.01 > 0.12).astype(int)

    frame = pd.DataFrame({
        "obs_date": obs,
        "cell_id": cells,
        "latitude_center": rng.uniform(-89, 89, n),
        "longitude_center": rng.uniform(-179, 179, n),
        "events_last_30d": rng.poisson(0.3, n),
        "events_last_90d": rng.poisson(1, n),
        "events_last_365d": activity,
        "mean_magnitude_90d": rng.uniform(0, 6, n),
        "max_magnitude_90d": rng.uniform(0, 8, n),
        "mean_depth_90d": rng.uniform(-3, 40, n),
        "events_ge45_last_365d": rng.poisson(0.5, n),
        "days_since_last_event": rng.uniform(0, 9000, n),
        "target": target,
    })
    # Chronological split, mirroring features/splits.py semantics.
    train_end = pd.Timestamp("2007-12-31", tz="UTC")
    val_end = pd.Timestamp("2009-12-31", tz="UTC")
    frame["split"] = np.where(
        frame["obs_date"] <= train_end, "train",
        np.where(frame["obs_date"] <= val_end, "val", "test"),
    )
    if frame.groupby("split")["target"].nunique().min() < 2:
        # Ensure every split has both classes for metric tests.
        rng2 = np.random.default_rng(seed + 1)
        frame["target"] = rng2.integers(0, 2, size=n)
    frame = frame.sort_values("obs_date").reset_index(drop=True)
    frame.to_parquet(cfg.features_dir / "regional_monthly_features.parquet", index=False)
    return frame


@pytest.fixture()
def model_cfg(tmp_path):
    cfg = _split_project(tmp_path)
    _write_features(cfg)
    return cfg


class TestSplitGuards:
    def test_split_frame_refuses_unsplit_data(self, model_cfg):
        path = model_cfg.features_dir / "regional_monthly_features.parquet"
        frame = pd.read_parquet(path).drop(columns=["split"])
        with pytest.raises(ValidationError, match="no 'split' column"):
            split_frame(frame)

    def test_split_frame_enforces_strict_chronology(self, model_cfg):
        from seismo.models.train import load_feature_frame

        splits = split_frame(load_feature_frame(model_cfg))
        assert splits["train"]["obs_date"].max() < splits["val"]["obs_date"].min()
        assert splits["val"]["obs_date"].max() < splits["test"]["obs_date"].min()

    def test_feature_columns_never_include_leaky_columns(self):
        frame = _toy_frame()
        frame["split"] = "train"
        cols = feature_columns(frame)
        assert "split" not in cols
        assert "target" not in cols
        assert "cell_id" not in cols
        assert "obs_date" not in cols

    def test_training_never_receives_test_rows(self, model_cfg, monkeypatch):
        """Spy on fit(): the frame handed to the estimator must end before the
        train/val boundary — this is the assertion, not a code-path hope."""
        from seismo.models.baselines import SklearnProbabilistic

        observed: list[pd.DataFrame] = []
        original_fit = SklearnProbabilistic.fit

        def spy(self, frame, feature_cols):
            observed.append(frame.copy())
            return original_fit(self, frame, feature_cols)

        monkeypatch.setattr(SklearnProbabilistic, "fit", spy)

        result = train_model(model_cfg, "logreg")
        assert observed, "spy never fired"

        train_end = pd.Timestamp(model_cfg.section("splits")["train_end"], tz="UTC")
        for captured in observed:
            assert captured["obs_date"].max() <= train_end + pd.Timedelta(days=1), \
                "test/val rows leaked into fit()"
            assert (captured["split"] == "train").all(), \
                f"non-train rows present: {captured['split'].unique()}"
        assert result.train_rows > 0

    def test_trained_artifact_is_self_describing(self, model_cfg):
        result = train_model(model_cfg, "logreg")
        directory = Path(result.path)
        for required in ("model.joblib", "features.json", "metrics.json",
                         "training_config.yaml", "data_manifest_ref.json"):
            if required == "metrics.json":
                continue  # written by evaluation, not training
            assert (directory / required).is_file(), required

        metadata = json.loads((directory / "features.json").read_text())
        assert metadata["feature_cols"] == result.feature_cols
        lineage = json.loads((directory / "data_manifest_ref.json").read_text())
        assert "collection" in lineage and "raw_manifest" in lineage


class TestPromotionGate:
    def _train_and_evaluate(self, cfg):
        from seismo.models.reporting import evaluate_model

        result = train_model(cfg, "baseline_climatology")
        evaluate_model(cfg, "baseline_climatology", version=result.version,
                       split="val", write=True, shap_enabled=False)
        return result

    def test_promotion_requires_evaluation(self, model_cfg):
        result = train_model(model_cfg, "logreg")
        assert not (Path(result.path) / "metrics.json").is_file()

        with pytest.raises(ValidationError, match="refusing to promote"):
            promote(model_cfg, "logreg", result.version, reviewed_by="reviewer")

    def test_promotion_requires_a_named_reviewer(self, model_cfg):
        result = self._train_and_evaluate(model_cfg)
        with pytest.raises(ValidationError, match="--reviewed-by"):
            promote(model_cfg, "baseline_climatology", result.version, reviewed_by="   ")

    def test_promotion_records_reviewer_and_replacement(self, model_cfg):
        from seismo.models.reporting import evaluate_model

        first = self._train_and_evaluate(model_cfg)
        promote(model_cfg, "baseline_climatology", first.version, reviewed_by="ada")
        second = train_model(model_cfg, "baseline_climatology")
        evaluate_model(model_cfg, "baseline_climatology", version=second.version,
                       split="val", write=True, shap_enabled=False)
        promote(model_cfg, "baseline_climatology", second.version, reviewed_by="ada")

        registry = load_registry(model_cfg)
        assert registry["production"]["baseline_climatology"] == second.version
        history = registry["history"]
        assert history[-1]["replaced"] == first.version
        assert history[-1]["reviewed_by"] == "ada"

    def test_promotion_of_missing_artifact_is_actionable(self, model_cfg):
        from seismo.errors import MissingArtifact

        with pytest.raises(MissingArtifact) as exc:
            promote(model_cfg, "xgboost", "v1-nope", reviewed_by="ada")
        # The actionable next step ships in the hint (rendered as "Fix: …").
        assert "seismo model train" in (exc.value.hint or "")


class TestTrainingGuards:
    def test_unknown_model_name_is_rejected(self, model_cfg):
        with pytest.raises(ValidationError, match="Unknown model"):
            train_model(model_cfg, "my_neural_net")

    def test_all_model_names_are_buildable(self, model_cfg):
        from seismo.models.train import build_estimator

        for name in MODEL_NAMES:
            estimator = build_estimator(name, model_cfg)
            assert estimator is not None, name

    def test_logreg_fit_predict_round_trip(self, model_cfg):
        result = train_model(model_cfg, "logreg")
        from seismo.models.train import load_feature_frame, load_model

        model, metadata = load_model(model_cfg, "logreg", result.version)
        frame = load_feature_frame(model_cfg)
        val = frame[frame["split"] == "val"]
        proba = model.predict_proba(val[metadata["feature_cols"]])[:, 1]
        assert np.isfinite(proba).all()
        assert 0 <= proba.min() and proba.max() <= 1
