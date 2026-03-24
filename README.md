# pyeddies

Boundary layer dynamics & turbulence post-processing toolkit for high-order CFD.

## Features
- Material properties: NASA-9 thermodynamics, Sutherland viscosity
- BL profiles: Reichardt + Coles wake, auto u_tau tuning
- Post-processing: Wall-normal profiles, Cf, delta_99, theta, H12
- Turbulence: Reynolds stress, Favre averaging, Van Driest transform
- Film cooling: eta effectiveness, BR verification, jet trajectory
- High-order interpolation: p=4 Lagrange with barycentric weights
- I/O: Gmsh to Fluent mesh conversion, CGNS reader

## Install

```bash
pip install -e .          # editable (development)
pip install -e ".[full]"  # with scipy, matplotlib
```

## Quick Start

```python
from pyeddies import FlowField, StreamwiseSweep
from pyeddies.material import get_air_nasa9

# Load tavg VTU
ff = FlowField("tavg_merged.vtu")

# Streamwise analysis
air = get_air_nasa9()
params = air.pyfr_constants(T_ref=537, Re_target=26000, L_ref=0.01, Me=0.3)
sweep = StreamwiseSweep(ff, x_stations=[0.01, 0.03, 0.05], flow_params=params)
sweep.compute_all()
sweep.tier2_summary()
```

## Documentation
See [docs/](docs/) for detailed guides and API reference.

## License
MIT
