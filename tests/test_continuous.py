"""Tests for continuous EM estimation."""
import numpy as np
import pytest

from emery import (
    estimate_ML,
    estimate_ML_continuous,
    generate_multimethod_continuous,
    generate_multimethod_data,
    pollinate_ML_continuous,
)
from emery.continuous import _dmvnorm


def _sim(seed=0, n_obs=200, n_method=2):
    np.random.seed(seed)
    return generate_multimethod_data(
        "continuous",
        n_obs=n_obs,
        n_method=n_method,
        mu_i1=[12.0] * n_method,
        mu_i0=[10.0] * n_method,
        sigma_i1=np.eye(n_method),
        sigma_i0=np.eye(n_method),
    )


class TestGenerateContinuous:
    def test_shape(self):
        sim = _sim()
        assert sim["generated_data"].shape == (200, 2)

    def test_params_keys(self):
        sim = _sim()
        for key in ("n_method", "n_obs", "prev", "D",
                    "mu_i1", "mu_i0", "sigma_i1", "sigma_i0"):
            assert key in sim["params"]

    def test_positive_group_higher_mean(self):
        """Positive observations should have higher mean (mu_i1=12 > mu_i0=10)."""
        sim = _sim(n_obs=1000)
        data = sim["generated_data"]
        D_arr = np.array(list(sim["params"]["D"].values()))
        mean_pos = np.nanmean(data[D_arr == 1, :], axis=0)
        mean_neg = np.nanmean(data[D_arr == 0, :], axis=0)
        assert np.all(mean_pos > mean_neg)


class TestDmvnorm:
    def test_scalar(self):
        x = np.array([[0.0, 0.0]])
        mu = np.array([0.0, 0.0])
        sigma = np.eye(2)
        d = _dmvnorm(x, mu, sigma)
        expected = 1.0 / (2 * np.pi)
        np.testing.assert_allclose(d[0], expected, rtol=1e-6)

    def test_shape(self):
        x = np.random.randn(50, 3)
        mu = np.zeros(3)
        sigma = np.eye(3)
        d = _dmvnorm(x, mu, sigma)
        assert d.shape == (50,)
        assert np.all(d > 0)

    def test_symmetric(self):
        """Density should be the same at x and -x for zero-mean symmetric dist."""
        x = np.array([[1.0, 2.0], [-1.0, -2.0]])
        mu = np.zeros(2)
        sigma = np.eye(2)
        d = _dmvnorm(x, mu, sigma)
        np.testing.assert_allclose(d[0], d[1], rtol=1e-10)


class TestPollinate:
    def test_keys(self):
        sim = _sim()
        init = pollinate_ML_continuous(sim["generated_data"])
        for key in ("prev_1", "mu_i1_1", "sigma_i1_1", "mu_i0_1", "sigma_i0_1"):
            assert key in init

    def test_high_pos_ordering(self):
        """With high_pos=True, mu_i1 (positive group) should be > mu_i0."""
        sim = _sim()
        init = pollinate_ML_continuous(sim["generated_data"], high_pos=True)
        assert np.all(init["mu_i1_1"] > init["mu_i0_1"])

    def test_sigma_positive_definite(self):
        sim = _sim()
        init = pollinate_ML_continuous(sim["generated_data"])
        eigs_1 = np.linalg.eigvalsh(init["sigma_i1_1"])
        eigs_0 = np.linalg.eigvalsh(init["sigma_i0_1"])
        assert np.all(eigs_1 > 0)
        assert np.all(eigs_0 > 0)


class TestEstimateContinuous:
    def setup_method(self):
        np.random.seed(42)
        sim = generate_multimethod_continuous(
            n_obs=300, n_method=2,
            mu_i1=[12.0, 12.0], mu_i0=[10.0, 10.0],
            sigma_i1=np.eye(2), sigma_i0=np.eye(2),
            prev=0.4,
        )
        self.data = sim["generated_data"]
        self.params = sim["params"]

    def test_type(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert result.type == "continuous"

    def test_result_keys(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        for key in ("prev_est", "mu_i1_est", "sigma_i1_est",
                    "mu_i0_est", "sigma_i0_est", "eta_j_est",
                    "A_j_est", "z_k1_est", "z_k0_est"):
            assert key in result.results

    def test_prev_reasonable(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert abs(result.results["prev_est"] - 0.4) < 0.15

    def test_mu_ordering(self):
        """Positive group mean should be higher than negative."""
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert np.all(result.results["mu_i1_est"] > result.results["mu_i0_est"])

    def test_auc_better_than_chance(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert np.all(result.results["A_j_est"] > 0.5)

    def test_auc_in_01(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        A = result.results["A_j_est"]
        assert np.all(A >= 0) and np.all(A <= 1)

    def test_z_k1_in_01(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        z = result.results["z_k1_est"]
        assert np.all(z >= 0) and np.all(z <= 1)

    def test_z_k1_plus_z_k0_equals_one(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        total = result.results["z_k1_est"] + result.results["z_k0_est"]
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_progress_shape(self):
        result = estimate_ML_continuous(self.data, save_progress=True)
        n_iter = result.iter
        n_obs = self.data.shape[0]
        assert result.prog["prev"].shape == (n_iter,)
        assert result.prog["z_k1"].shape == (n_iter, n_obs)

    def test_no_progress_when_disabled(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert result.prog == {}

    def test_dispatch(self):
        init = pollinate_ML_continuous(self.data)
        r1 = estimate_ML_continuous(self.data, init=init, save_progress=False)
        r2 = estimate_ML("continuous", data=self.data, init=init, save_progress=False)
        np.testing.assert_allclose(r1.results["prev_est"], r2.results["prev_est"], rtol=1e-3)

    def test_repr_hides_posteriors(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        rep = repr(result)
        assert "z_k1_est" not in rep
        assert "z_k0_est" not in rep
        assert "prev_est" in rep
        assert "A_j_est" in rep

    def test_sigma_positive_definite(self):
        result = estimate_ML_continuous(self.data, save_progress=False)
        eigs_1 = np.linalg.eigvalsh(result.results["sigma_i1_est"])
        eigs_0 = np.linalg.eigvalsh(result.results["sigma_i0_est"])
        assert np.all(eigs_1 > 0)
        assert np.all(eigs_0 > 0)

    def test_eta_j_sign(self):
        """eta_j = (mu1 - mu0) / std should be positive when mu1 > mu0."""
        result = estimate_ML_continuous(self.data, save_progress=False)
        assert np.all(result.results["eta_j_est"] > 0)
