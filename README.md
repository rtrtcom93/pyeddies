# pyeddies

Boundary layer dynamics & turbulence post-processing toolkit for high-order CFD.

## Features
- **FlowParams**: params.yaml one-line load, auto Phase 1-4 computation
- **Material properties**: NASA-9 thermodynamics, Sutherland viscosity, correlations
- **BL profiles**: Reichardt + Coles wake, auto u_tau tuning
- **Post-processing**: Wall-normal profiles, Cf, delta_99, theta, H12
- **Turbulence**: Reynolds stress, Favre averaging, Van Driest transform
- **Film cooling**: eta effectiveness, BR verification, jet trajectory (stub)
- **Snapshot analysis**: Q-criterion, lambda2, vorticity, slicing
- **Time series**: PSD, Strouhal, autocorrelation, SPOD/DMD input
- **Visualization**: profile plots, 2D contours, 3D isosurfaces
- **High-order interpolation**: p=4 Lagrange with barycentric weights

## Install

```bash
pip install -e .          # editable (development)
pip install -e ".[full]"  # with scipy, matplotlib
```

## Quick Start

```python
from pyeddies import FlowParams, FlowField

# (A) Parameters only — no VTU needed
fp = FlowParams.from_yaml("params.yaml")
print(fp.pe, fp.utau_0, fp.Taw)

# (B) VTU + params — full analysis
ff = FlowField("tavg.vtu", params="params.yaml")
sweep = ff.sweep([0.03, 0.05, 0.07])

# (C) Slicing + visualization
s = ff.slice('z')                    # z = z_mid (auto)
fig, ax = s.plot('avg-T')           # matplotlib contour
df = s.to_dataframe()               # pandas DataFrame

# (D) Region clipping (chaining)
ff.box(x=[0, 0.1]).slice('y', 0.0).plot('avg-u')
```

## Documentation
See [docs/](docs/) for detailed guides and API reference.

## License
MIT
