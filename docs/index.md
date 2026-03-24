# pyeddies Documentation

Boundary layer dynamics & turbulence post-processing toolkit for high-order CFD.

## Package Structure

```
pyeddies/
├── params.py       — FlowParams: YAML → auto-computed flow parameters (Phase 1-4)
├── material/       — NASA-9 thermodynamics, Sutherland viscosity, correlations
├── profile/        — Reichardt + Coles wake BL profiles, tune_utau
├── post/           — VTU post-processing pipeline
│   ├── core.py     — FlowField (VTU loader, slice/box/probe, SliceResult)
│   ├── interp.py   — p=4 Lagrange barycentric interpolation
│   ├── wall.py     — tau_w, Cf, u_tau, wall units
│   ├── integral.py — delta_99, theta, delta*, H12, Re_theta, Re_tau
│   ├── transforms.py — Van Driest, Crocco-Busemann, semi-local scaling
│   ├── turbstats.py  — Reynolds stress, Favre averaging, TKE
│   ├── sweep.py    — StreamwiseSweep (multi-station CTBL analysis)
│   ├── fc.py       — FilmCoolingSweep (η, BR, jet trajectory — stub)
│   ├── snapshot.py — SnapshotAnalyzer (Q-criterion, λ2, vorticity)
│   └── timeseries.py — ProbeTimeSeries (PSD), PlaneTimeSeries (SPOD input)
├── viz/            — Visualization (matplotlib + optional pyvista)
│   ├── profiles.py — u+(y+), η(x/D), Cf(x), Reynolds stress, PSD
│   ├── contours.py — 2D slice contour (tricontourf)
│   └── three_d.py  — 3D isosurface (pyvista)
├── modal/          — [placeholder] SPOD, DMD
└── io/             — [placeholder] Gmsh→Fluent, CGNS utilities
```

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `FlowParams` | `pyeddies.params` | params.yaml → auto Phase 1-4 계산 |
| `FlowField` | `pyeddies.post.core` | VTU 로드 + slice/box/probe + sweep/fc_sweep |
| `SliceResult` | `pyeddies.post.core` | slice 결과 → .plot(), .to_dataframe() |
| `StreamwiseSweep` | `pyeddies.post.sweep` | 다중 x-station CTBL 분석 |
| `FilmCoolingSweep` | `pyeddies.post.fc` | FC η, BR (stub, tavg 데이터 대기) |
| `SnapshotAnalyzer` | `pyeddies.post.snapshot` | Q-criterion, λ2, slicing |
| `SnapshotSequence` | `pyeddies.post.snapshot` | 다중 snapshot 시계열 추출 |
| `ProbeTimeSeries` | `pyeddies.post.timeseries` | PSD, Strouhal, 적분시간스케일 |
| `PlaneTimeSeries` | `pyeddies.post.timeseries` | SPOD/DMD snapshot matrix |

## Guides

1. [Installation](installation.md)
2. [Getting Started](getting-started.md)
3. Detailed guides — `guides/` (planned)

## API Reference

See `api/` directory (planned).

## Theory Background

See `theory/` for derivations and references (planned).
