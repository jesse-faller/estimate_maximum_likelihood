"""
Tests for binary EM estimation, mirroring the R testthat suite.
"""
import numpy as np
import pytest

from emery import (
    bin_auc,
    estimate_ML,
    estimate_ML_binary,
    generate_multimethod_data,
    pollinate_ML_binary,
    random_start_binary,
    unique_obs_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_INIT = {"prev_1": 0.2, "se_1": np.full(4, 0.75), "sp_1": np.full(4, 0.75)}


def _generate_fixed_data(seed=0):
    np.random.seed(seed)
    sim = generate_multimethod_data(
        "binary",
        n_obs=75,
        n_method=4,
        se=[0.87, 0.92, 0.79, 0.95],
        sp=[0.85, 0.93, 0.94, 0.80],
        prev=0.5,
    )
    return sim["generated_data"], sim["params"]


# ---------------------------------------------------------------------------
# Test: full data vs compressed data give identical estimates
# ---------------------------------------------------------------------------

class TestCompressedDataConsistency:
    def setup_method(self):
        data, _ = _generate_fixed_data(seed=123)
        self.data = data
        self.init = _FIXED_INIT.copy()

    def test_prev_est_matches(self):
        result_full = estimate_ML_binary(
            self.data, init=self.init, tol=1e-7, max_iter=1000, save_progress=False
        )
        summary = unique_obs_summary(self.data)
        result_fast = estimate_ML_binary(
            summary["unique_obs"],
            freqs=summary["obs_freqs"],
            init=self.init,
            tol=1e-7,
            max_iter=1000,
            save_progress=False,
        )
        np.testing.assert_allclose(
            result_full.results["prev_est"],
            result_fast.results["prev_est"],
            rtol=1e-6,
        )

    def test_se_est_matches(self):
        result_full = estimate_ML_binary(
            self.data, init=self.init, tol=1e-7, max_iter=1000, save_progress=False
        )
        summary = unique_obs_summary(self.data)
        result_fast = estimate_ML_binary(
            summary["unique_obs"],
            freqs=summary["obs_freqs"],
            init=self.init,
            tol=1e-7,
            max_iter=1000,
            save_progress=False,
        )
        np.testing.assert_allclose(
            result_full.results["se_est"],
            result_fast.results["se_est"],
            rtol=1e-6,
        )

    def test_sp_est_matches(self):
        result_full = estimate_ML_binary(
            self.data, init=self.init, tol=1e-7, max_iter=1000, save_progress=False
        )
        summary = unique_obs_summary(self.data)
        result_fast = estimate_ML_binary(
            summary["unique_obs"],
            freqs=summary["obs_freqs"],
            init=self.init,
            tol=1e-7,
            max_iter=1000,
            save_progress=False,
        )
        np.testing.assert_allclose(
            result_full.results["sp_est"],
            result_fast.results["sp_est"],
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Test: estimates are reasonable given known true parameters
# ---------------------------------------------------------------------------

class TestEstimationQuality:
    def setup_method(self):
        # Larger sample for reliable estimates
        np.random.seed(7)
        sim = generate_multimethod_data(
            "binary",
            n_obs=500,
            n_method=3,
            se=[0.90, 0.85, 0.80],
            sp=[0.90, 0.88, 0.85],
            prev=0.4,
        )
        self.data = sim["generated_data"]
        self.params = sim["params"]

    def test_prev_in_reasonable_range(self):
        result = estimate_ML_binary(self.data, tol=1e-7, save_progress=False)
        # With 500 obs, estimate should be within ±0.1 of 0.4
        assert abs(result.results["prev_est"] - 0.4) < 0.1

    def test_se_in_reasonable_range(self):
        result = estimate_ML_binary(self.data, tol=1e-7, save_progress=False)
        true_se = np.array([0.90, 0.85, 0.80])
        np.testing.assert_allclose(result.results["se_est"], true_se, atol=0.15)

    def test_sp_in_reasonable_range(self):
        result = estimate_ML_binary(self.data, tol=1e-7, save_progress=False)
        true_sp = np.array([0.90, 0.88, 0.85])
        np.testing.assert_allclose(result.results["sp_est"], true_sp, atol=0.15)

    def test_qk_sum_equals_prevalence(self):
        """sum(q_k) / n_obs should equal the estimated prevalence."""
        result = estimate_ML_binary(self.data, tol=1e-7, save_progress=False)
        qk_mean = np.mean(result.results["qk_est"])
        assert abs(qk_mean - result.results["prev_est"]) < 1e-6

    def test_qk_in_01(self):
        result = estimate_ML_binary(self.data, tol=1e-7, save_progress=False)
        assert np.all(result.results["qk_est"] >= 0.0)
        assert np.all(result.results["qk_est"] <= 1.0)


# ---------------------------------------------------------------------------
# Test: convergence and iteration count
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_iterations_stored(self):
        data, _ = _generate_fixed_data()
        result = estimate_ML_binary(data, save_progress=True)
        assert result.iter > 0
        assert result.iter <= 1000

    def test_progress_shape(self):
        data, _ = _generate_fixed_data()
        n_obs, n_method = data.shape
        result = estimate_ML_binary(data, save_progress=True)
        n_iter = result.iter
        assert result.prog["prev"].shape == (n_iter,)
        assert result.prog["se"].shape == (n_iter, n_method)
        assert result.prog["sp"].shape == (n_iter, n_method)
        assert result.prog["qk"].shape == (n_iter, n_obs)

    def test_no_progress_when_disabled(self):
        data, _ = _generate_fixed_data()
        result = estimate_ML_binary(data, save_progress=False)
        assert result.prog == {}

    def test_tighter_tol_more_iterations(self):
        data, _ = _generate_fixed_data()
        r_loose = estimate_ML_binary(data, tol=1e-3, save_progress=False)
        r_tight = estimate_ML_binary(data, tol=1e-9, save_progress=False)
        assert r_tight.iter >= r_loose.iter


# ---------------------------------------------------------------------------
# Test: missing data (NaN) handling
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_estimate_with_some_nan(self):
        np.random.seed(1)
        sim = generate_multimethod_data(
            "binary",
            n_obs=50,
            n_method=3,
            n_method_subset=2,   # only 2 of 3 observed per row
        )
        data = sim["generated_data"]
        assert np.any(np.isnan(data)), "Expected some NaN values"
        result = estimate_ML_binary(data, save_progress=False)
        assert 0 < result.results["prev_est"] < 1
        assert np.all(result.results["se_est"] > 0)
        assert np.all(result.results["sp_est"] > 0)

    def test_compressed_with_nan(self):
        np.random.seed(2)
        sim = generate_multimethod_data(
            "binary", n_obs=40, n_method=3, n_method_subset=2
        )
        data = sim["generated_data"]
        init = {"prev_1": 0.3, "se_1": np.full(3, 0.8), "sp_1": np.full(3, 0.8)}
        result_full = estimate_ML_binary(data, init=init, save_progress=False)
        summary = unique_obs_summary(data)
        result_fast = estimate_ML_binary(
            summary["unique_obs"],
            freqs=summary["obs_freqs"],
            init=init,
            save_progress=False,
        )
        np.testing.assert_allclose(
            result_full.results["prev_est"],
            result_fast.results["prev_est"],
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Test: initialisation functions
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_pollinate_keys(self):
        data, _ = _generate_fixed_data()
        init = pollinate_ML_binary(data)
        assert set(init.keys()) == {"prev_1", "se_1", "sp_1"}

    def test_pollinate_prev_in_01(self):
        data, _ = _generate_fixed_data()
        init = pollinate_ML_binary(data)
        assert 0 < init["prev_1"] < 1

    def test_pollinate_se_sp_in_01(self):
        data, _ = _generate_fixed_data()
        init = pollinate_ML_binary(data)
        assert np.all((init["se_1"] > 0) & (init["se_1"] < 1))
        assert np.all((init["sp_1"] > 0) & (init["sp_1"] < 1))

    def test_random_start_keys(self):
        init = random_start_binary(n_method=4)
        assert set(init.keys()) == {"prev_1", "se_1", "sp_1"}
        assert len(init["se_1"]) == 4

    def test_random_start_better_than_chance(self):
        for _ in range(20):
            init = random_start_binary(n_method=3)
            # se + sp >= 1 for all methods
            assert np.all(init["se_1"] + init["sp_1"] >= 1.0)


# ---------------------------------------------------------------------------
# Test: bin_auc
# ---------------------------------------------------------------------------

class TestBinAuc:
    def test_perfect_classifier(self):
        auc = bin_auc(1.0, 1.0)
        np.testing.assert_allclose(auc, 1.0)

    def test_random_classifier(self):
        # se=0.5, sp=0.5 → AUC = 0.5
        auc = bin_auc(0.5, 0.5)
        np.testing.assert_allclose(auc, 0.5)

    def test_vectorised(self):
        auc = bin_auc([0.9, 0.8], [0.85, 0.75])
        assert len(auc) == 2
        assert np.all((auc > 0) & (auc <= 1))

    def test_known_value(self):
        # se=0.9, sp=0.9 → AUC = 0.9
        auc = bin_auc(0.9, 0.9)
        np.testing.assert_allclose(auc, 0.9)


# ---------------------------------------------------------------------------
# Test: unique_obs_summary
# ---------------------------------------------------------------------------

class TestUniqueObsSummary:
    def test_no_duplicates(self):
        data = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=float)
        result = unique_obs_summary(data)
        assert len(result["unique_obs"]) == 3
        assert np.all(result["obs_freqs"] == 1)

    def test_with_duplicates(self):
        data = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        result = unique_obs_summary(data)
        assert len(result["unique_obs"]) == 2
        freqs = sorted(result["obs_freqs"])
        assert freqs == [1, 2]

    def test_with_nan(self):
        data = np.array([[1, np.nan], [1, np.nan], [0, 1]], dtype=float)
        result = unique_obs_summary(data)
        assert len(result["unique_obs"]) == 2
        assert np.sum(result["obs_freqs"]) == 3

    def test_total_frequency_preserved(self):
        data, _ = _generate_fixed_data()
        result = unique_obs_summary(data)
        assert np.sum(result["obs_freqs"]) == len(data)
