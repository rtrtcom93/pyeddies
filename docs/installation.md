# Installation

## Requirements

- Python >= 3.10
- numpy >= 1.24
- pyvista >= 0.40 (VTU loading)
- h5py >= 3.0
- pyyaml >= 6.0

Optional:
- scipy >= 1.10 (Lagrange interpolation cKDTree)
- matplotlib >= 3.7 (plotting)

## Editable Install (Development)

```bash
cd pyeddies
pip install -e .
# or with all optional deps:
pip install -e ".[full]"
```

## From fc-pyfr-iles-project

```bash
cd fc-pyfr-iles-project
pip install -e ../pyeddies
```

## Verify

```python
from pyeddies import FlowParams, __version__
print(f"pyeddies {__version__}")

fp = FlowParams()
print(f"Default CTBL: pe={fp.pe:.0f} Pa, utau={fp.utau_0:.3f} m/s")
```
