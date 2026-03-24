"""Tests for pyeddies.material — thermodynamics, transport, correlations."""
import numpy as np
import pytest

from pyeddies.material import (
    get_air_nasa9, mu_sutherland, SpecificHeat,
    AIR_PARK, PropertyTable, MaterialData,
    cf_smits, re_tau_schlatter, van_driest_ii_adiabatic,
    H12_chauhan,
)


class TestNASA9:
    """NASA-9 thermodynamic properties."""

    def setup_method(self):
        self.air = get_air_nasa9()

    def test_cp_537K(self):
        """Cp(537K) ≈ 1036.94 J/(kg·K) — CLAUDE.md reference."""
        assert self.air.Cp(537.0) == pytest.approx(1036.94, rel=1e-3)

    def test_gamma_537K(self):
        """gamma(537K) ≈ 1.3827 — CLAUDE.md reference."""
        assert self.air.gamma(537.0) == pytest.approx(1.3827, rel=1e-3)

    def test_gamma_range(self):
        """gamma monotonically decreases with T for air (300-2000K)."""
        T = np.linspace(300, 2000, 50)
        g = np.array([self.air.gamma(t) for t in T])
        assert np.all(np.diff(g) < 0)

    def test_cp_positive(self):
        """Cp should be positive for all reasonable T."""
        for T in [200, 300, 537, 1000, 2000, 5000]:
            assert self.air.Cp(T) > 0

    def test_region_continuity(self):
        """Cp continuous across NASA-9 region boundary (1000K)."""
        cp_lo = self.air.Cp(999.9)
        cp_hi = self.air.Cp(1000.1)
        assert abs(cp_hi - cp_lo) / cp_lo < 0.01


class TestSutherland:
    """Sutherland viscosity law."""

    def test_mu_537K(self):
        """mu(537K) ≈ 2.802e-5 — CLAUDE.md reference."""
        assert mu_sutherland(537.0) == pytest.approx(2.802e-5, rel=1e-2)

    def test_mu_increasing(self):
        """Viscosity increases with temperature."""
        assert mu_sutherland(300) < mu_sutherland(537) < mu_sutherland(1000)

    def test_mu_positive(self):
        for T in [200, 300, 537, 1000, 2000]:
            assert mu_sutherland(T) > 0


class TestCorrelations:
    """Empirical BL correlations."""

    def test_cf_smits_decreasing(self):
        """Cf decreases with Re_theta."""
        Re = np.array([1000, 1400, 2000])
        cf = np.array([cf_smits(r) for r in Re])
        assert np.all(np.diff(cf) < 0)

    def test_cf_smits_order(self):
        """Cf ~ O(1e-3) for typical Re_theta."""
        cf = cf_smits(1400)
        assert 1e-4 < cf < 1e-2

    def test_re_tau_schlatter(self):
        """Re_tau(1400) ≈ 507 — notebook reference."""
        assert re_tau_schlatter(1400) == pytest.approx(507, rel=0.02)

    def test_h12_chauhan(self):
        """H12(1400) ≈ 1.18 — turbulent BL."""
        h12 = H12_chauhan(1400)
        assert 1.1 < h12 < 1.6

    def test_van_driest_reduces_cf(self):
        """Van Driest-II: compressible Cf < incompressible Cf for Me>0."""
        cf_inc = 0.004
        cf_comp, Fc = van_driest_ii_adiabatic(cf_inc, Me=0.3)
        assert cf_comp < cf_inc
        assert Fc > 1.0


class TestDatabase:
    """Material property tables."""

    def test_air_park_has_cp(self):
        assert 'Cp' in AIR_PARK.properties

    def test_air_park_interp(self):
        """Interpolation returns reasonable Cp."""
        cp = AIR_PARK.get("Cp", 537.0)
        assert 900 < cp < 1200
