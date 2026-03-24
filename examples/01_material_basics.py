"""
pyeddies 기초: 공기 물성 계산
실행: python examples/01_material_basics.py
"""
import numpy as np
from pyeddies.material import get_air_nasa9, mu_sutherland

air = get_air_nasa9()
T = 537.0  # K

cp = air.Cp(T)
gamma = air.gamma(T)
mu = mu_sutherland(T)
R = cp * (gamma - 1) / gamma

print(f"Air at {T} K:")
print(f"  Cp    = {cp:.2f} J/(kg·K)")
print(f"  gamma = {gamma:.4f}")
print(f"  mu    = {mu:.4e} Pa·s")
print(f"  R     = {R:.3f} J/(kg·K)")

# Temperature sweep
print("\n--- Temperature sweep ---")
T_range = np.array([200, 300, 500, 537, 800, 1000, 1500, 2000])
print(f"{'T [K]':>8s} {'Cp [J/kg/K]':>12s} {'gamma':>8s} {'mu [Pa·s]':>12s}")
for T_i in T_range:
    print(f"{T_i:8.0f} {air.Cp(T_i):12.2f} {air.gamma(T_i):8.4f} {mu_sutherland(T_i):12.4e}")
