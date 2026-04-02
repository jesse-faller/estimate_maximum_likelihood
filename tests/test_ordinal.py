"""Tests for ordinal EM estimation."""
import numpy as np
import pytest

from emery import (
    estimate_ML,
    estimate_ML_ordinal,
    generate_multimethod_data,
    generate_multimethod_ordinal,
    pollinate_ML_ordinal,
    unique_obs_summary,
)


def _sim(seed=0, n_obs=200, n_method=3, n_level=5):
    np.random.seed(seed)
    return generate_multimethod_data(
        "ordinal", n_obs=n_obs, n_method=n_method, n_level=n_level
    )


class TestGenerateOrdinal:
    def test_shape(self):
        sim = _sim()
        data = sim["generated_data"]
        assert data.shape == (200, 3)

    def test_levels_in_range(self):
        sim = _sim()
        data = sim["generated_data"]
        valid = data[~np.isnan(data)]
        assert np.all(valid >= 1) and np.all(valid <= 5)

    def test_params_keys(self):
        sim = _sim()
        for key in ("n_method", "n_level", "n_obs", "prev", "D",
                    "pmf_pos", "pmf_neg", "method_names", "level_names"):
            assert key in sim["params"]

    def test_pmf_normalised(self):
        sim = _sim()
        np.testing.assert_allclose(sim["params"]["pmf_pos"].sum(axis=1), 1.0)
        np.testing.assert_allclose(sim["params"]["pmf_neg"].sum(axis=1), 1.0)


class TestPollinate:
    def test_keys(self):
        sim = _sim()
        init = pollinate_ML_ordinal(sim["generated_data"])
        assert set(init.keys()) == {"pi_1_1", "phi_1ij_1", "phi_0ij_1", "n_level"}

    def test_prev_in_01(self):
        sim = _sim()
        init = pollinate_ML_ordinal(sim["generated_data"])
        assert 0 < init["pi_1_1"] < 1

    def test_phi_sums_approx_one(self):
        sim = _sim()
        init = pollinate_ML_ordinal(sim["generated_data"])
        # Each column of phi should sum to approximately 1
        col_sums_1 = init["phi_1ij_1"].sum(axis=0)
        col_sums_0 = init["phi_0ij_1"].sum(axis=0)
        np.testing.assert_allclose(col_sums_1, 1.0, atol=0.05)
        np.testing.assert_allclose(col_sums_0, 1.0, atol=0.05)


class TestEstimateOrdinal:
    def setup_method(self):
        np.random.seed(1)
        sim = generate_multimethod_data(
            "ordinal", n_obs=300, n_method=3, n_level=5, prev=0.4
        )
        self.data = sim["generated_data"]
        self.params = sim["params"]

    def test_returns_correct_type(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        assert result.type == "ordinal"

    def test_result_keys(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        for key in ("prev_est", "A_i_est", "phi_1ij_est", "phi_0ij_est", "q_k1_est"):
            assert key in result.results

    def test_prev_in_reasonable_range(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        assert abs(result.results["prev_est"] - 0.4) < 0.15

    def test_auc_in_01(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        A = result.results["A_i_est"]
        assert np.all(A >= 0) and np.all(A <= 1)

    def test_auc_better_than_chance(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        assert np.all(result.results["A_i_est"] > 0.5)

    def test_phi_shapes(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        n_level = len(result.names["level_names"])
        n_method = len(result.names["method_names"])
        assert result.results["phi_1ij_est"].shape == (n_level, n_method)
        assert result.results["phi_0ij_est"].shape == (n_level, n_method)

    def test_qk1_in_01(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        qk1 = result.results["q_k1_est"]
        assert np.all(qk1 >= 0) and np.all(qk1 <= 1)

    def test_qk1_sum_equals_prevalence(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        np.testing.assert_allclose(
            np.mean(result.results["q_k1_est"]),
            result.results["prev_est"],
            atol=1e-5,
        )

    def test_progress_shape(self):
        result = estimate_ML_ordinal(self.data, save_progress=True)
        n_iter = result.iter
        n_obs = self.data.shape[0]
        assert result.prog["prev"].shape == (n_iter,)
        assert result.prog["q_k1"].shape == (n_iter, n_obs)

    def test_no_progress_when_disabled(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        assert result.prog == {}

    def test_dispatch(self):
        r1 = estimate_ML_ordinal(self.data, save_progress=False)
        r2 = estimate_ML("ordinal", data=self.data, save_progress=False)
        np.testing.assert_allclose(r1.results["prev_est"], r2.results["prev_est"])

    def test_convergence_uses_likelihood(self):
        """Tighter tolerance should use more iterations."""
        r_loose = estimate_ML_ordinal(self.data, tol=1e-3, save_progress=False)
        r_tight = estimate_ML_ordinal(self.data, tol=1e-9, save_progress=False)
        assert r_tight.iter >= r_loose.iter

    def test_repr_hides_posteriors(self):
        result = estimate_ML_ordinal(self.data, save_progress=False)
        rep = repr(result)
        assert "q_k1_est" not in rep
        assert "prev_est" in rep


class TestOrdinalMissingData:
    def test_with_missing(self):
        np.random.seed(3)
        sim = generate_multimethod_ordinal(
            n_obs=100, n_method=3, n_level=5, n_method_subset=2
        )
        data = sim["generated_data"]
        assert np.any(np.isnan(data))
        result = estimate_ML_ordinal(data, save_progress=False)
        assert 0 < result.results["prev_est"] < 1
