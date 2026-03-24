"""
FlowParams: params.yaml → 전체 유동 파라미터 자동 계산
실행: python examples/03_flowparams.py
"""
from pyeddies import FlowParams

# --- 1. Default CTBL (no YAML needed) ---
print("=== Default CTBL ===")
fp = FlowParams()
print(fp)
print(f"  nu_w = {fp.nu_w:.4e}")
print(f"  dnu  = {fp.dnu_0:.4e}")

# --- 2. Custom FC parameters ---
print("\n=== Custom FC (BR=1.0, Tc=290K) ===")
fp_fc = FlowParams(
    Re_D=13000,
    D=0.01,
    d99_inlet=5.0e-3,
    Tc=290.0,
    BR=1.0,
)
print(fp_fc)

# --- 3. from_yaml (if params.yaml available) ---
import os
yaml_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'fc-pyfr-iles-project',
    'cases', 'ctbl_ma03_p4', 'params.yaml'
)
if os.path.isfile(yaml_path):
    print(f"\n=== from_yaml: {os.path.basename(yaml_path)} ===")
    fp_yaml = FlowParams.from_yaml(yaml_path)
    print(fp_yaml)
else:
    print(f"\n(params.yaml not found at {yaml_path}, skipping)")

# --- 4. to_dict for backward compatibility ---
print("\n=== to_dict keys ===")
d = fp_fc.to_dict()
print(f"  Keys: {sorted(d.keys())}")
