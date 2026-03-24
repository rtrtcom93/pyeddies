"""Tests for pyeddies.profile — mean BL profiles and parameter determination."""
import numpy as np
import pytest

from pyeddies.profile import (
    reichardt_uplus, coles_wake, tune_utau_for_delta99,
    u_prof_pyfr, mean_profile_wall_wake,
    delta99_from_profile, theta_momentum,
)


class TestReichardt:
    """Reichardt inner-law profile."""

    def test_viscous_sublayer(self):
        """u+ ≈ y+ in viscous sublayer (y+ < 3)."""
        yp = np.array([0.5, 1.0, 2.0])
        up = reichardt_uplus(yp)
        # Reichardt deviates slightly from u+=y+ even at low y+
        assert np.allclose(up, yp, atol=0.2)

    def test_log_layer(self):
        """u+ follows log-law in log layer (y+ ~ 30-300)."""
        yp = np.array([30, 100, 300])
        up = reichardt_uplus(yp)
        kappa, B = 0.41, 5.2
        up_log = (1 / kappa) * np.log(yp) + B
        assert np.allclose(up, up_log, rtol=0.05)

    def test_monotonic(self):
        """u+ is monotonically increasing."""
        yp = np.logspace(-1, 4, 500)
        up = reichardt_uplus(yp)
        assert np.all(np.diff(up) > 0)


class TestColesWake:
    """Coles wake function."""

    def test_zero_at_wall(self):
        """Wake = 0 at eta=0."""
        assert coles_wake(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_max_at_edge(self):
        """Wake = 2*sin^2(pi/2) = 2.0 at eta=1 (Pi_w applied externally)."""
        w = coles_wake(1.0)
        assert w == pytest.approx(2.0, rel=1e-3)

    def test_clipping(self):
        """eta > 1 should be clipped."""
        assert coles_wake(1.5) == coles_wake(1.0)
        assert coles_wake(-0.5) == coles_wake(0.0)


class TestTuneUtau:
    """Self-consistent u_tau determination."""

    def test_ctbl_utau(self):
        """CTBL d99=4.2mm → u_tau ≈ 6.45 m/s (CLAUDE.md reference)."""
        nu_w = 2.80237e-5 / 0.52612  # approximate
        u_tau = tune_utau_for_delta99(
            target_delta99=4.2e-3, ue=138.489, nuw=nu_w,
            kappa=0.41, B=5.2, Pi=0.45
        )
        assert u_tau == pytest.approx(6.45, rel=0.05)

    def test_fc_utau(self):
        """FC d99=5mm → u_tau ≈ 6.82 m/s (CLAUDE.md reference)."""
        nu_w = 2.80237e-5 / (40543 / (287.003 * 545.25))  # approximate
        u_tau = tune_utau_for_delta99(
            target_delta99=5.0e-3, ue=138.489, nuw=nu_w,
            kappa=0.41, B=5.2, Pi=0.45
        )
        assert u_tau == pytest.approx(6.82, rel=0.05)


class TestMeanProfile:
    """Full mean profile (velocity + temperature)."""

    def test_wall_no_slip(self):
        """u(y=0) ≈ 0."""
        y = np.linspace(0, 5e-3, 100)
        u, T, rho = mean_profile_wall_wake(
            y, ue=138.489, utau=6.45, nuw=5.33e-5
        )
        assert u[0] == pytest.approx(0.0, abs=1e-6)

    def test_freestream_recovery(self):
        """u(y >> d99) → ue."""
        y = np.linspace(0, 0.02, 200)
        u, T, rho = mean_profile_wall_wake(
            y, ue=138.489, utau=6.45, nuw=5.33e-5
        )
        assert u[-1] == pytest.approx(138.489, rel=0.01)


class TestProfileIntegrals:
    """BL integral quantities from profiles."""

    def test_delta99_blasius_like(self):
        """d99 detection for a smooth profile."""
        y = np.linspace(0, 10e-3, 500)
        ue = 100.0
        u = ue * np.tanh(y / 3e-3)
        d99 = delta99_from_profile(y, u, ue)
        assert 2e-3 < d99 < 10e-3

    def test_theta_positive(self):
        """Momentum thickness is positive for BL profiles."""
        y = np.linspace(0, 10e-3, 500)
        ue = 100.0
        u = ue * np.tanh(y / 3e-3)
        theta = theta_momentum(y, u, ue)
        assert theta > 0
