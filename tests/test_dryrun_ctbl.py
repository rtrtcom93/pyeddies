"""Dry-run integration test — CTBL Ma=0.3 P=4 actual data.

Uses real params.yaml, profile CSVs, and summary.csv from
fc-pyfr-iles-project/cases/ctbl_ma03_p4/.

Validates that pyeddies can reproduce the full post-processing pipeline
without VTU (CSV-based path).
"""
import os
import numpy as np
import pandas as pd
import pytest
import yaml

# -------------------------------------------------------
# Data paths
# -------------------------------------------------------
CASES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'fc-pyfr-iles-project', 'cases'
)
CTBL_DIR = os.path.join(CASES_DIR, 'ctbl_ma03_p4')

# Skip entire module if case data not found
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CTBL_DIR),
    reason="CTBL case data not found"
)


@pytest.fixture(scope="module")
def params():
    with open(os.path.join(CTBL_DIR, 'params.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(
        os.path.join(CTBL_DIR, 'results', 'statistics', 'summary.csv')
    )


@pytest.fixture(scope="module")
def profile_50mm():
    return pd.read_csv(
        os.path.join(CTBL_DIR, 'results', 'statistics', 'profiles',
                     'profile_x50mm.csv')
    )


# -------------------------------------------------------
# 1. Material: params.yaml 값 재현
# -------------------------------------------------------
class TestMaterialFromParams:
    """pyeddies.material로 params.yaml의 derived 값을 재현."""

    def test_gamma(self, params):
        from pyeddies.material import get_air_nasa9
        air = get_air_nasa9()
        Te = params['flow']['Te']
        gamma_computed = air.gamma(Te)
        gamma_ref = params['derived']['gamma']
        assert gamma_computed == pytest.approx(gamma_ref, rel=1e-4)

    def test_cp(self, params):
        from pyeddies.material import get_air_nasa9
        air = get_air_nasa9()
        Te = params['flow']['Te']
        cp_computed = air.Cp(Te)
        cp_ref = params['derived']['cp']
        assert cp_computed == pytest.approx(cp_ref, rel=1e-3)

    def test_mu(self, params):
        from pyeddies.material import mu_sutherland
        Te = params['flow']['Te']
        mu_computed = mu_sutherland(Te)
        mu_ref = params['derived']['mu_e']
        assert mu_computed == pytest.approx(mu_ref, rel=1e-3)

    def test_pe_from_Re(self, params):
        """pe = Re_D * mu_e * ue / (D * R * Te) — back-calculated."""
        drv = params['derived']
        fl = params['flow']
        re = params['reynolds']

        # rho_e = Re_D * mu / (ue * D)
        rho_e = re['Re_D'] * drv['mu_e'] / (drv['ue'] * re['D'])
        pe = rho_e * drv['R_specific'] * fl['Te']
        assert pe == pytest.approx(drv['pe'], rel=1e-3)


# -------------------------------------------------------
# 2. Profile: u_tau from params.yaml
# -------------------------------------------------------
class TestProfileFromParams:
    """pyeddies.profile로 params.yaml의 u_tau를 재현."""

    def test_tune_utau(self, params):
        from pyeddies.profile import tune_utau_for_delta99
        bl = params['boundary_layer']
        drv = params['derived']

        u_tau = tune_utau_for_delta99(
            target_delta99=bl['d99_inlet'],
            ue=drv['ue'],
            nuw=drv['nu_w'],
            kappa=bl['kappa'],
            B=bl['B'],
            Pi=bl['Pi_wake'],
        )
        assert u_tau == pytest.approx(drv['utau_inlet'], rel=0.02)

    def test_reichardt_viscous_sublayer(self, params):
        """u+ ≈ y+ for y+ < 3 at CTBL conditions."""
        from pyeddies.profile import reichardt_uplus
        yp = np.array([0.5, 1.0, 2.0])
        up = reichardt_uplus(yp)
        assert np.allclose(up, yp, atol=0.2)


# -------------------------------------------------------
# 3. Post: BL integrals from actual profile CSV
# -------------------------------------------------------
class TestPostFromCSV:
    """실제 profile CSV로 BL 적분량 계산."""

    def test_delta99(self, profile_50mm, params):
        from pyeddies.post import delta_99
        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        ue = params['derived']['ue']

        d99 = delta_99(y, u, u_e=ue)
        # Should be close to ~4.2mm (inlet) or slightly grown
        assert 3.0e-3 < d99 < 6.0e-3

    def test_momentum_thickness(self, profile_50mm, params):
        from pyeddies.post.integral import momentum_thickness
        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        ue = params['derived']['ue']

        theta = momentum_thickness(y, u, ue)
        # Typical: 0.4~0.6 mm
        assert 0.3e-3 < theta < 1.0e-3

    def test_displacement_thickness(self, profile_50mm, params):
        from pyeddies.post.integral import displacement_thickness
        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        ue = params['derived']['ue']

        ds = displacement_thickness(y, u, ue)
        # Typical: 0.6~1.0 mm
        assert 0.4e-3 < ds < 1.5e-3

    def test_shape_factor(self, profile_50mm, params):
        from pyeddies.post.integral import (
            momentum_thickness, displacement_thickness, shape_factor,
        )
        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        ue = params['derived']['ue']

        theta = momentum_thickness(y, u, ue)
        ds = displacement_thickness(y, u, ue)
        H12 = shape_factor(ds, theta)
        # Turbulent BL: 1.2 ~ 1.6
        assert 1.1 < H12 < 1.8

    def test_bl_integrals_all(self, profile_50mm, params):
        from pyeddies.post.integral import compute_bl_integrals
        drv = params['derived']
        y = profile_50mm['y'].values
        u = profile_50mm['u'].values

        result = compute_bl_integrals(
            y, u, u_e=drv['ue'], nu_e=drv['nu_e'],
            nu_w=drv['nu_w'], u_tau=drv['utau_inlet'],
        )
        assert 'delta_99' in result
        assert 'theta' in result
        assert 'Re_theta' in result
        assert result['Re_theta'] > 1000


# -------------------------------------------------------
# 4. Correlations: summary.csv 검증
# -------------------------------------------------------
class TestCorrelationsFromSummary:
    """summary.csv의 Cf, Re_theta를 상관식과 비교."""

    def test_cf_vs_smits(self, summary):
        """Cf vs Smits correlation — order of magnitude match."""
        from pyeddies.material import cf_smits

        # x >= 40mm (SEM recovery 이후)
        mask = summary['x_mm'] >= 40
        for _, row in summary[mask].iterrows():
            cf_corr = cf_smits(row['Re_theta'])
            # Allow factor 2 (ILES vs empirical)
            assert 0.3 * cf_corr < row['Cf'] < 3.0 * cf_corr

    def test_re_tau_vs_schlatter(self, summary):
        """Re_tau vs Schlatter correlation."""
        from pyeddies.material import re_tau_schlatter

        mask = summary['x_mm'] >= 40
        for _, row in summary[mask].iterrows():
            re_tau_corr = re_tau_schlatter(row['Re_theta'])
            # Allow 30% deviation
            assert row['Re_tau'] == pytest.approx(re_tau_corr, rel=0.30)

    def test_h12_vs_chauhan(self, summary):
        """H12 vs Chauhan correlation — order-of-magnitude.

        Note: Chauhan is incompressible; ILES compressible H12 ~30% higher.
        """
        from pyeddies.material import H12_chauhan

        mask = summary['x_mm'] >= 40
        for _, row in summary[mask].iterrows():
            h12_corr = H12_chauhan(row['Re_theta'])
            # Compressible H12 is higher; allow 40%
            assert row['H12'] == pytest.approx(h12_corr, rel=0.40)

    def test_streamwise_growth(self, summary):
        """delta_99 grows downstream (monotonic after SEM recovery)."""
        mask = summary['x_mm'] >= 40
        d99 = summary[mask]['delta_99_mm'].values
        # Allow small non-monotonicity, but trend should be positive
        assert d99[-1] > d99[0]

    def test_re_theta_grows(self, summary):
        """Re_theta grows downstream."""
        mask = summary['x_mm'] >= 40
        re_th = summary[mask]['Re_theta'].values
        assert re_th[-1] > re_th[0]


# -------------------------------------------------------
# 5. Wall properties from profile CSV
# -------------------------------------------------------
class TestWallFromCSV:
    """wall.py 함수들을 실제 프로파일로 테스트."""

    def test_tau_w_from_profile(self, profile_50mm, params):
        from pyeddies.post.wall import compute_tau_w, mu_sutherland

        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        T_w = profile_50mm['T'].values[0]
        mu_w = mu_sutherland(T_w)

        # Use first few wall-adjacent points
        n_wall = min(10, len(y))
        tau_w, dUdy_w = compute_tau_w(y[:n_wall], u[:n_wall], mu_w)
        # Positive, reasonable magnitude
        assert tau_w > 0
        assert 5.0 < tau_w < 50.0

    def test_cf_from_profile(self, profile_50mm, params):
        from pyeddies.post.wall import compute_tau_w, compute_cf, mu_sutherland
        drv = params['derived']

        y = profile_50mm['y'].values
        u = profile_50mm['u'].values
        T_w = profile_50mm['T'].values[0]
        mu_w = mu_sutherland(T_w)

        n_wall = min(10, len(y))
        tau_w, _ = compute_tau_w(y[:n_wall], u[:n_wall], mu_w)
        cf = compute_cf(tau_w, drv['rho_e'], drv['ue'])
        # Cf ~ O(1e-3) for turbulent BL
        assert 1e-4 < cf < 1e-2


# -------------------------------------------------------
# 6. FlowField VTU load (optional — heavy)
# -------------------------------------------------------
VTU_PATH = os.path.join(
    CTBL_DIR, 'results', 'statistics', 'vtu',
    'ctbl-ma03-0.00932-p4-stats.vtu'
)


@pytest.mark.skipif(
    not os.path.isfile(VTU_PATH),
    reason="VTU file not found (large file)"
)
class TestFlowFieldVTU:
    """FlowField 로드 + 기본 필드 접근 (VTU 존재 시만)."""

    def test_load(self):
        from pyeddies.post.core import FlowField
        ff = FlowField(VTU_PATH)
        assert ff.mesh.n_points > 0
        assert ff.mesh.n_cells > 0

    def test_mode_detection(self):
        from pyeddies.post.core import FlowField
        ff = FlowField(VTU_PATH)
        assert ff.mode == 'tavg'

    def test_field_access(self):
        from pyeddies.post.core import FlowField
        ff = FlowField(VTU_PATH)
        fields = ff.available_fields()
        assert 'u' in fields
        assert 'rho' in fields
        assert 'T' in fields
