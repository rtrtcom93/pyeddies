"""
경계층 속도 프로파일: Reichardt + Coles wake + tune_utau
실행: python examples/02_bl_profile.py
"""
import numpy as np
from pyeddies.profile import reichardt_uplus, coles_wake, tune_utau_for_delta99
from pyeddies.profile import mean_profile_wall_wake

# --- 1. Reichardt inner law ---
yp = np.logspace(-1, 3, 200)
up = reichardt_uplus(yp)
print("Reichardt u+(y+) — viscous sublayer check:")
print(f"  y+=1.0 → u+={reichardt_uplus(1.0):.3f} (should be ~1.0)")
print(f"  y+=100 → u+={reichardt_uplus(100.0):.3f}")
print(f"  y+=1000 → u+={reichardt_uplus(1000.0):.3f}")

# --- 2. Coles wake function ---
eta = np.linspace(0, 1, 5)
wake_vals = coles_wake(eta)
print(f"\nColes wake W(eta): {dict(zip(eta, wake_vals))}")
print(f"  W(0)=0, W(1)=2 (maximum)")

# --- 3. tune_utau for CTBL ---
print("\n--- tune_utau_for_delta99 ---")
u_tau = tune_utau_for_delta99(
    target_delta99=4.2e-3,   # 4.2 mm
    ue=138.489,              # m/s
    nuw=5.469e-5,            # nu_w (wall kinematic viscosity)
    kappa=0.41, B=5.2, Pi=0.45,
)
print(f"  d99 = 4.2 mm, ue = 138.49 m/s")
print(f"  → u_tau = {u_tau:.3f} m/s")

# --- 4. Full profile (mean_profile_wall_wake) ---
nu_w = 5.469e-5
y = np.linspace(0, 10e-3, 500)
u, T, rho = mean_profile_wall_wake(
    y, ue=138.489, utau=u_tau, nuw=nu_w,
    kappa=0.41, B=5.2, Pi=0.45, delta=4.2e-3,
)
print(f"\nMean profile (incompressible):")
print(f"  u(0) = {u[0]:.6f} (no-slip)")
print(f"  u(d99) ≈ {u[np.searchsorted(y, 4.2e-3)]:.3f} m/s")
print(f"  u(max) = {u[-1]:.3f} m/s (ue = 138.489)")
