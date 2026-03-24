"""Tests for pyeddies.params — FlowParams dataclass."""
import os
import pytest
import yaml

from pyeddies.params import FlowParams


# -------------------------------------------------------
# Data paths
# -------------------------------------------------------
CASES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'fc-pyfr-iles-project', 'cases'
)
CTBL_YAML = os.path.join(CASES_DIR, 'ctbl_ma03_p4', 'params.yaml')
FC_YAML = os.path.join(CASES_DIR, 'fc_sr_br1p0_ma03_p4', 'params.yaml')


# -------------------------------------------------------
# 1. Basic construction (no YAML needed)
# -------------------------------------------------------
class TestFlowParamsBasic:
    """FlowParams without YAML — default CTBL values."""

    def test_default_construction(self):
        fp = FlowParams()
        assert fp.gamma > 1.3
        assert fp.pe > 0
        assert fp.utau_0 > 0
        assert fp.dnu_0 > 0

    def test_thermodynamic_consistency(self):
        fp = FlowParams()
        # R = cp * (gamma-1) / gamma
        R_check = fp.cp * (fp.gamma - 1) / fp.gamma
        assert R_check == pytest.approx(fp.R_gas, rel=1e-6)

    def test_pe_from_re(self):
        fp = FlowParams()
        # rho_e = Re_D * mu_e / (ue * D)
        rho_check = fp.Re_D * fp.mu_e / (fp.ue * fp.D)
        assert rho_check == pytest.approx(fp.rhoe, rel=1e-6)

    def test_to_dict(self):
        fp = FlowParams()
        d = fp.to_dict()
        assert 'pe' in d
        assert 'rhoe' in d
        assert 'rho_e' in d  # alias
        assert 'utau_inlet' in d  # alias
        assert d['d99_0'] == fp.d99_inlet

    def test_fc_not_computed_without_tc(self):
        fp = FlowParams()
        assert fp.rhoc is None
        assert fp.DR is None


class TestFlowParamsFC:
    """FlowParams with FC parameters."""

    def test_fc_construction(self):
        fp = FlowParams(Tc=290.0, BR=1.0, Re_D=13000, d99_inlet=0.005)
        assert fp.DR is not None
        assert fp.DR > 1.0  # cold coolant → denser
        assert fp.VR is not None
        assert fp.Uc is not None

    def test_dr_formula(self):
        fp = FlowParams(Tc=290.0, BR=1.0, Re_D=13000, d99_inlet=0.005)
        dr_check = fp.rhoc / fp.rhoe
        assert dr_check == pytest.approx(fp.DR, rel=1e-10)

    def test_vr_formula(self):
        fp = FlowParams(Tc=290.0, BR=1.0, Re_D=13000, d99_inlet=0.005)
        vr_check = fp.BR / fp.DR
        assert vr_check == pytest.approx(fp.VR, rel=1e-10)


# -------------------------------------------------------
# 2. from_yaml — real data (skip if not found)
# -------------------------------------------------------
@pytest.mark.skipif(
    not os.path.isfile(CTBL_YAML),
    reason="CTBL params.yaml not found"
)
class TestFlowParamsCTBLYaml:
    """FlowParams.from_yaml with CTBL params.yaml."""

    def test_load(self):
        fp = FlowParams.from_yaml(CTBL_YAML)
        assert fp.case_name == 'ctbl_ma03_p4'

    def test_pe_matches(self):
        fp = FlowParams.from_yaml(CTBL_YAML)
        with open(CTBL_YAML, encoding='utf-8') as f:
            ref = yaml.safe_load(f)['derived']
        assert fp.pe == pytest.approx(ref['pe'], rel=1e-3)

    def test_rhoe_matches(self):
        fp = FlowParams.from_yaml(CTBL_YAML)
        with open(CTBL_YAML, encoding='utf-8') as f:
            ref = yaml.safe_load(f)['derived']
        assert fp.rhoe == pytest.approx(ref['rho_e'], rel=1e-3)

    def test_utau_matches(self):
        fp = FlowParams.from_yaml(CTBL_YAML)
        with open(CTBL_YAML, encoding='utf-8') as f:
            ref = yaml.safe_load(f)['derived']
        assert fp.utau_0 == pytest.approx(ref['utau_inlet'], rel=0.02)

    def test_repr(self):
        fp = FlowParams.from_yaml(CTBL_YAML)
        r = repr(fp)
        assert 'ctbl' in r
        assert 'pe=' in r


@pytest.mark.skipif(
    not os.path.isfile(FC_YAML),
    reason="FC params.yaml not found"
)
class TestFlowParamsFCYaml:
    """FlowParams.from_yaml with FC params.yaml."""

    def test_load(self):
        fp = FlowParams.from_yaml(FC_YAML)
        assert fp.Tc is not None

    def test_dr_matches(self):
        fp = FlowParams.from_yaml(FC_YAML)
        with open(FC_YAML, encoding='utf-8') as f:
            ref = yaml.safe_load(f)['derived']
        assert fp.DR == pytest.approx(ref['DR'], rel=1e-3)

    def test_pe_matches(self):
        fp = FlowParams.from_yaml(FC_YAML)
        with open(FC_YAML, encoding='utf-8') as f:
            ref = yaml.safe_load(f)['derived']
        assert fp.pe == pytest.approx(ref['pe'], rel=1e-3)

    def test_to_dict_fc_keys(self):
        fp = FlowParams.from_yaml(FC_YAML)
        d = fp.to_dict()
        assert 'Tc' in d
        assert 'DR' in d
        assert 'rho_c' in d  # alias
