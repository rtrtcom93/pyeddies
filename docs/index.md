# pyeddies Documentation

Boundary layer dynamics & turbulence post-processing toolkit for high-order CFD.

## Package Structure

```
pyeddies/
├── params.py       — FlowParams: YAML → auto-computed flow parameters
├── material/       — NASA-9 thermodynamics, Sutherland viscosity, correlations
├── profile/        — Reichardt + Coles wake BL profiles, tune_utau
├── post/           — VTU post-processing pipeline
│   ├── core.py     — FlowField (VTU loader, tavg/snapshot detection)
│   ├── interp.py   — p=4 Lagrange barycentric interpolation
│   ├── wall.py     — tau_w, Cf, u_tau, wall units
│   ├── integral.py — delta_99, theta, delta*, H12, Re_theta, Re_tau
│   ├── transforms.py — Van Driest, Crocco-Busemann, semi-local scaling
│   ├── turbstats.py  — Reynolds stress, Favre averaging, TKE
│   ├── sweep.py    — StreamwiseSweep (multi-station CTBL analysis)
│   ├── fc.py       — FilmCoolingSweep (η, BR, jet trajectory)
│   ├── snapshot.py — SnapshotAnalyzer (Q-criterion, λ2, vorticity)
│   └── timeseries.py — ProbeTimeSeries (PSD), PlaneTimeSeries (SPOD input)
├── modal/          — [placeholder] SPOD, DMD
└── io/             — [placeholder] Gmsh→Fluent, CGNS utilities
```

## Guides

1. [Installation](installation.md)
2. [Getting Started](getting-started.md)
3. Detailed guides — `guides/`

## API Reference

See `api/` directory for module-level documentation.

## Theory Background

See `theory/` for derivations and references.
