"""Dry-run integration test — FC Single-Row BR=1.0 Ma=0.3.

Uses real params.yaml from fc-pyfr-iles-project/cases/fc_sr_br1p0_ma03_p4/.
FC 계산이 아직 진행 중이므로 VTU/profile 없이 파라미터 검증만 수행.
"""
import os
import numpy as np
import pytest
import yaml

# -------------------------------------------------------
# Data paths
# -------------------------------------------------------
CASES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'fc-pyfr-iles-project', 'cases'
)
FC_DIR = os.path.join(CASES_DIR, 'fc_sr_br1p0_ma03_p4')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(FC_DIR),
    reason="FC case data not found"
)


@pytest.fixture(scope="module")
def params():
    with open(os.path.join(FC_DIR, 'params.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# 1. Material: 열역학 일관성 (Tier 1-A)
# -------------------------------------------------------
class TestThermodynamicConsistency:
    """pyeddies.material로 FC params.yaml의 열역학 값 재현."""

    def test_gamma(self, params):
        from pyeddies.material import get_air_nasa9
        air = get_air_nasa9()
        gamma = air.gamma(params['flow']['Te'])
        assert gamma == pytest.approx(params['derived']['gamma'], rel=1e-4)

    def test_cp(self, params):
        from pyeddies.material import get_air_nasa9
        air = get_air_nasa9()
        cp = air.Cp(params['flow']['Te'])
        assert cp == pytest.approx(params['derived']['cp'], rel=1e-3)

    def test_R_specific(self, params):
        """R = Cp * (gamma-1) / gamma."""
        drv = params['derived']
        R_calc = drv['cp'] * (drv['gamma'] - 1) / drv['gamma']
        assert R_calc == pytest.approx(drv['R_specific'], rel=1e-3)

    def test_sound_speed(self, params):
        """a = sqrt(gamma * R * Te)."""
        drv = params['derived']
        fl = params['flow']
        a_calc = np.sqrt(drv['gamma'] * drv['R_specific'] * fl['Te'])
        assert a_calc == pytest.approx(drv['a'], rel=1e-3)

    def test_ue(self, params):
        """ue = Me * a."""
        drv = params['derived']
        fl = params['flow']
        ue_calc = fl['Me'] * drv['a']
        assert ue_calc == pytest.approx(drv['ue'], rel=1e-3)

    def test_mu(self, params):
        from pyeddies.material import mu_sutherland
        mu = mu_sutherland(params['flow']['Te'])
        assert mu == pytest.approx(params['derived']['mu_e'], rel=1e-3)


# -------------------------------------------------------
# 2. pe 역산 (Tier 1-A 연장)
# -------------------------------------------------------
class TestPeBackCalculation:
    """Re_D → pe 역산 검증."""

    def test_rho_e(self, params):
        """rho_e = Re_D * mu_e / (ue * D)."""
        drv = params['derived']
        re = params['reynolds']
        rho_e = re['Re_D'] * drv['mu_e'] / (drv['ue'] * re['D'])
        assert rho_e == pytest.approx(drv['rho_e'], rel=1e-3)

    def test_pe(self, params):
        """pe = rho_e * R * Te."""
        drv = params['derived']
        fl = params['flow']
        pe = drv['rho_e'] * drv['R_specific'] * fl['Te']
        assert pe == pytest.approx(drv['pe'], rel=1e-3)

    def test_nu_e(self, params):
        """nu_e = mu_e / rho_e."""
        drv = params['derived']
        nu_e = drv['mu_e'] / drv['rho_e']
        assert nu_e == pytest.approx(drv['nu_e'], rel=1e-3)


# -------------------------------------------------------
# 3. Wall properties (Tier 1-A: isentropic relations)
# -------------------------------------------------------
class TestWallProperties:
    """Taw, rho_w, mu_w, nu_w 검증."""

    def test_taw(self, params):
        """Taw = Te * (1 + r*(gamma-1)/2*Me^2), r=Pr^(1/3)."""
        fl = params['flow']
        drv = params['derived']
        r = fl['Pr'] ** (1.0 / 3.0)
        Taw = fl['Te'] * (1 + r * (drv['gamma'] - 1) / 2 * fl['Me'] ** 2)
        assert Taw == pytest.approx(drv['Taw'], rel=1e-3)

    def test_rho_w(self, params):
        """rho_w = pe / (R * Taw)."""
        drv = params['derived']
        rho_w = drv['pe'] / (drv['R_specific'] * drv['Taw'])
        assert rho_w == pytest.approx(drv['rho_w'], rel=1e-3)

    def test_mu_w(self, params):
        from pyeddies.material import mu_sutherland
        mu_w = mu_sutherland(params['derived']['Taw'])
        assert mu_w == pytest.approx(params['derived']['mu_w'], rel=1e-3)

    def test_nu_w(self, params):
        drv = params['derived']
        nu_w = drv['mu_w'] / drv['rho_w']
        assert nu_w == pytest.approx(drv['nu_w'], rel=1e-3)


# -------------------------------------------------------
# 4. Friction velocity (Tier 1-A: self-consistent u_tau)
# -------------------------------------------------------
class TestFrictionVelocity:
    """tune_utau_for_delta99 검증."""

    def test_utau_inlet(self, params):
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


# -------------------------------------------------------
# 5. Coolant properties (Tier 1-B: BR 정합성)
# -------------------------------------------------------
class TestCoolantProperties:
    """FC 쿨런트 물성 검증."""

    def test_rho_c(self, params):
        """rho_c = pe / (R * Tc)."""
        drv = params['derived']
        fc = params['filmcool']
        rho_c = drv['pe'] / (drv['R_specific'] * fc['Tc'])
        assert rho_c == pytest.approx(drv['rho_c'], rel=1e-3)

    def test_density_ratio(self, params):
        """DR = rho_c / rho_e."""
        drv = params['derived']
        DR = drv['rho_c'] / drv['rho_e']
        assert DR == pytest.approx(drv['DR'], rel=1e-3)

    def test_DR_greater_than_1(self, params):
        """Cold coolant → DR > 1 (denser)."""
        assert params['derived']['DR'] > 1.0

    def test_uc_br05(self, params):
        """Uc = BR * ue * rho_e / rho_c (BR=0.5)."""
        drv = params['derived']
        Uc = 0.5 * drv['ue'] * drv['rho_e'] / drv['rho_c']
        # Allow larger tolerance — Uc sensitive to rounding
        assert Uc == pytest.approx(drv['Uc_br05'], rel=0.02)

    def test_VR_br05(self, params):
        """VR = Uc / ue = BR / DR."""
        drv = params['derived']
        VR = 0.5 / drv['DR']
        assert VR == pytest.approx(drv['VR_br05'], rel=0.02)

    def test_momentum_flux_ratio(self, params):
        """I = BR * VR = BR^2 / DR."""
        drv = params['derived']
        I = 0.5 * drv['VR_br05']
        assert I == pytest.approx(drv['I_br05'], rel=0.02)


# -------------------------------------------------------
# 6. gamma 상수 가정 검증 (Tier 1-F)
# -------------------------------------------------------
class TestGammaAssumption:
    """gamma(Te) vs gamma(Tc) 차이가 작은지 확인."""

    def test_delta_gamma(self, params):
        from pyeddies.material import get_air_nasa9
        air = get_air_nasa9()
        fc = params['filmcool']
        fl = params['flow']

        gamma_e = air.gamma(fl['Te'])
        gamma_c = air.gamma(fc['Tc'])
        delta_pct = abs(gamma_e - gamma_c) / gamma_e * 100

        # Should be < 2% (calorically perfect gas assumption valid)
        assert delta_pct < 2.0
        assert delta_pct == pytest.approx(
            params['derived']['delta_gamma_pct'], rel=0.1
        )
