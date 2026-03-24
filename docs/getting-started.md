# Getting Started

## 3 Modes of Use

### Mode A: Parameters Only (No VTU)

```python
from pyeddies import FlowParams

# From params.yaml
fp = FlowParams.from_yaml("cases/ctbl_ma03_p4/params.yaml")
print(fp.pe)       # 81086 Pa
print(fp.utau_0)   # 6.452 m/s
print(fp.Taw)      # 545.25 K

# Or construct directly
fp = FlowParams(Te=537.0, Me=0.3, Re_D=26000, d99_inlet=4.2e-3)
```

### Mode B: Quick VTU Visualization (No params)

```python
from pyeddies import FlowField

ff = FlowField("tavg.vtu")
print(ff.mode)              # 'tavg'
print(ff.available_fields()) # ['p', 'rho', 'u', 'v', 'w', ...]
```

### Mode C: Full Analysis (VTU + params)

```python
from pyeddies import FlowField

ff = FlowField("tavg.vtu", params="params.yaml")
sweep = ff.sweep([0.01, 0.03, 0.05])
# sweep.compute_all()
# sweep.tier2_summary()
```

## Slicing, Clipping & Probing

```python
from pyeddies import FlowField

ff = FlowField("tavg.vtu", params="params.yaml")

# Slice by axis name (auto mid-plane)
s = ff.slice('z')                    # z = z_mid
s = ff.slice('z', 0.015)            # z = 0.015 m
s = ff.slice('x', x_over_D=5)       # x = 5D (requires params)

# SliceResult — plot & export
fig, ax = s.plot('avg-T')           # matplotlib tricontourf
fig, ax = s.plot('avg-u', cmap='viridis', levels=80)
df = s.to_dataframe()               # pandas DataFrame
print(s.n_points, s.array_names)

# Box clipping (chaining)
ff.box(x=[0, 0.1]).slice('y', 0.001).plot('avg-u')
ff.box(x=[0.02, 0.08], y=[0, 0.01]).slice('z').plot('avg-T')

# Point probing
result = ff.probe([[0.05, 0.001, 0.015]])
print(result)  # dict of arrays
```

## Visualization

```python
from pyeddies.viz import plot_uplus_yplus, plot_cf_x, plot_psd
from pyeddies.viz import plot_slice_contour

# Profile plots (from sweep results)
fig, ax = plot_uplus_yplus(yplus, uplus, label="x/D=5")
fig, ax = plot_cf_x(x_stations, cf_values)

# PSD plot
fig, ax = plot_psd(freq, Pxx, xlabel="f [Hz]")

# 2D contour from PyVista slice
fig, ax = plot_slice_contour(slice_mesh, 'avg-T', cmap='hot')

# 3D isosurface (requires pyvista)
from pyeddies.viz import plot_isosurface
pl = plot_isosurface(mesh, 'Q_criterion', 1e6, color='cyan')
pl.show()
```

## Material Properties

```python
from pyeddies.material import get_air_nasa9, mu_sutherland

air = get_air_nasa9()
print(air.Cp(537.0))      # 1036.94 J/(kg·K)
print(air.gamma(537.0))   # 1.3827
print(mu_sutherland(537.0)) # 2.802e-5 Pa·s
```

## BL Profile Construction

```python
from pyeddies.profile import tune_utau_for_delta99, mean_profile_wall_wake
import numpy as np

u_tau = tune_utau_for_delta99(
    target_delta99=4.2e-3, ue=138.489, nuw=5.47e-5
)

y = np.linspace(0, 10e-3, 500)
u, T, rho = mean_profile_wall_wake(
    y, ue=138.489, utau=u_tau, nuw=5.47e-5, delta=4.2e-3
)
```

## Correlations

```python
from pyeddies.material import cf_smits, re_tau_schlatter, H12_chauhan

Re_theta = 1400
print(f"Cf = {cf_smits(Re_theta):.5f}")
print(f"Re_tau = {re_tau_schlatter(Re_theta):.0f}")
print(f"H12 = {H12_chauhan(Re_theta):.3f}")
```

## Snapshot Analysis

```python
from pyeddies.post import SnapshotAnalyzer, SnapshotSequence

# Single snapshot
ff = FlowField("snapshot.vtu", params="params.yaml")
sa = ff.snapshot_analysis()
Q = sa.compute_q_criterion()
lam2 = sa.compute_lambda2()
plane = sa.slice_z()  # centerline

# Multiple snapshots
seq = SnapshotSequence("results/inst/fc-*.vtu", flow_params={...})
probe_data = seq.extract_probe_timeseries(points=[[0.05, 0.001, 0.015]])
```

## Time Series Analysis

```python
from pyeddies.post import ProbeTimeSeries

pts = ProbeTimeSeries(t=t_array, data={'u': u_array, 'v': v_array})
f, Pxx = pts.psd('u', nperseg=512)
St, Pxx_st = pts.strouhal('u', D=0.01, U_ref=138.49)
T_int = pts.integral_time_scale('u')
```
