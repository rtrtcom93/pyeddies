"""Tests for pyeddies.post — post-processing (non-pyvista parts)."""
import numpy as np
import pytest

from pyeddies.post import (
    mu_sutherland, compute_tau_w, compute_u_tau, compute_cf,
    wall_units,
    reynolds_stress, turbulence_intensity,
    van_driest_transform, crocco_busemann, semi_local_scaling,
    delta_99, momentum_thickness, displacement_thickness,
    shape_factor, re_theta, re_tau, compute_bl_integrals,
)


class TestWallProperties:
    """Wall shear stress and friction velocity."""

    def test_mu_sutherland_consistency(self):
        """Post's mu_sutherland matches material's."""
        from pyeddies.material import mu_sutherland as mu_mat
        T = 537.0
        assert mu_sutherland(T) == pytest.approx(mu_mat(T), rel=1e-10)

    def test_utau_from_tauw(self):
        """u_tau = sqrt(tau_w / rho_w)."""
        tau_w = 20.0
        rho_w = 0.5
        u_tau = compute_u_tau(tau_w, rho_w)
        assert u_tau == pytest.approx(np.sqrt(tau_w / rho_w), rel=1e-10)

    def test_cf_definition(self):
        """Cf = 2 * tau_w / (rho_e * ue^2)."""
        tau_w, rho_e, ue = 20.0, 0.526, 138.489
        cf = compute_cf(tau_w, rho_e, ue)
        expected = 2.0 * tau_w / (rho_e * ue**2)
        assert cf == pytest.approx(expected, rel=1e-10)

    def test_wall_units_scaling(self):
        """y+ = y * u_tau / nu_w, u+ = u / u_tau."""
        y = np.array([0, 1e-5, 1e-4])
        u = np.array([0, 1.0, 10.0])
        u_tau = 6.0
        nu_w = 5e-5
        result = wall_units(y, u, u_tau, nu_w)
        assert np.allclose(result['y_plus'], y * u_tau / nu_w)
        assert np.allclose(result['u_plus'], u / u_tau)


class TestTransforms:
    """Compressibility transformations."""

    def test_van_driest_incompressible_limit(self):
        """Uniform density → VD transform is identity."""
        n = 50
        yp = np.linspace(0, 500, n)
        up = np.linspace(0, 20, n)
        rho = np.ones(n)
        rho_w = 1.0
        up_vd = van_driest_transform(yp, up, rho, rho_w)
        assert np.allclose(up_vd, up, rtol=1e-10)

    def test_crocco_busemann_wall(self):
        """T(u=0) = T_w (wall temperature)."""
        u = np.linspace(0, 138.489, 100)
        ue = 138.489
        Te = 537.0
        T = crocco_busemann(u, ue, Te, gamma=1.3827, Me=0.3, Pr=0.71)
        # T at wall should be Taw (adiabatic wall)
        assert T[0] > Te  # wall is hotter than freestream


class TestIntegrals:
    """BL integral quantities."""

    def test_d99_top_hat(self):
        """Top-hat profile: d99 = 0 (all at ue)."""
        y = np.linspace(0, 10e-3, 100)
        u = np.ones_like(y) * 100.0
        d = delta_99(y, u, u_e=100.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_displacement_thickness_positive(self):
        """delta* > 0 for BL profile."""
        y = np.linspace(0, 10e-3, 500)
        ue = 100.0
        u = ue * np.tanh(y / 3e-3)
        ds = displacement_thickness(y, u, ue)
        assert ds > 0

    def test_shape_factor_range(self):
        """H12 = delta*/theta, typically 1.3-2.6 for turbulent BL."""
        h12 = shape_factor(delta_star=1.0e-3, theta=0.7e-3)
        assert 1.0 < h12 < 3.0

    def test_re_theta_positive(self):
        assert re_theta(0.5e-3, 138.0, 5e-5) > 0

    def test_re_tau_positive(self):
        assert re_tau(6.0, 4.2e-3, 5e-5) > 0

    def test_bl_integrals_dict(self):
        """compute_bl_integrals returns expected keys."""
        y = np.linspace(0, 10e-3, 500)
        ue = 100.0
        u = ue * np.tanh(y / 3e-3)
        result = compute_bl_integrals(y, u, u_e=ue, nu_e=5e-5,
                                      nu_w=5e-5, u_tau=6.0)
        for key in ['delta_99', 'theta', 'delta_star', 'H12']:
            assert key in result
