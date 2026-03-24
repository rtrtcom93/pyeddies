"""core.py — VTU load, field access, slicing.

Unified access layer for instantaneous snapshots and time-averaged (tavg) fields.
"""
import os

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


# -------------------------------------------------------
# Field name mapping tables
# -------------------------------------------------------

_SNAP_MAP = {
    'rho': ('Density', None),
    'u':   ('Velocity', 0),
    'v':   ('Velocity', 1),
    'w':   ('Velocity', 2),
    'p':   ('Pressure', None),
}

_TAVG_PREFIXES = ['avg-', 'avg_', 'Avg-', 'Avg_']

_TAVG_FIELDS = [
    'rho', 'u', 'v', 'w', 'p', 'T',
    'uu', 'vv', 'ww', 'uv', 'uw', 'vw',
    'rhou', 'rhov', 'rhow',
    'rhouu', 'rhovv', 'rhoww', 'rhouv', 'rhouw', 'rhovw',
    'TT', 'uT', 'vT', 'wT',
]


def _find_tavg_field(array_names, base_name):
    """Try multiple naming conventions for a tavg field."""
    for prefix in _TAVG_PREFIXES:
        candidate = prefix + base_name
        if candidate in array_names:
            return candidate
    for name in array_names:
        if name.lower().replace('-', '').replace('_', '') == ('avg' + base_name).lower():
            return name
    return None


class FlowField:
    """VTU file wrapper with automatic snapshot/tavg detection."""

    def __init__(self, vtu_path):
        if pv is None:
            raise ImportError("pyvista is required for FlowField. Install: pip install pyvista")
        if not os.path.exists(vtu_path):
            raise FileNotFoundError(f"VTU not found: {vtu_path}")
        self.path = vtu_path
        self.mesh = pv.read(vtu_path)
        self.mode = self._detect_mode()
        self._field_map = self._build_field_map()

    def _detect_mode(self):
        names = set(self.mesh.array_names)
        for prefix in _TAVG_PREFIXES:
            if any(n.startswith(prefix) for n in names):
                return 'tavg'
        if 'Density' in names or 'Velocity' in names or 'Pressure' in names:
            return 'instantaneous'
        return 'unknown'

    def _build_field_map(self):
        fmap = {}
        names = set(self.mesh.array_names)

        if self.mode == 'instantaneous':
            for norm_name, (vtu_name, comp) in _SNAP_MAP.items():
                if vtu_name in names:
                    fmap[norm_name] = (vtu_name, comp)
        elif self.mode == 'tavg':
            for base in _TAVG_FIELDS:
                vtu_name = _find_tavg_field(names, base)
                if vtu_name is not None:
                    fmap[base] = (vtu_name, None)
        return fmap

    def get(self, name, R_gas=287.0):
        """Unified field access by normalized name.

        Special: 'T' is computed from p/(rho*R_gas) if not directly available.
        """
        if name == 'T' and 'T' not in self._field_map:
            p = self.get('p')
            rho = self.get('rho')
            return p / (rho * R_gas)

        if name not in self._field_map:
            raise KeyError(
                f"Field '{name}' not available. "
                f"Mode={self.mode}, available: {sorted(self._field_map.keys())}"
            )
        vtu_name, comp = self._field_map[name]
        data = self.mesh.point_data[vtu_name]
        if comp is not None:
            return data[:, comp]
        return data

    def slice(self, origin, normal):
        """PyVista slice."""
        return self.mesh.slice(normal=normal, origin=origin)

    def slice_clip(self, origin, normal, axis='y', vmin=None, vmax=None):
        """Slice + clip to a window along an axis.

        Preserved from original PyFRStats._slice_clip_window.
        """
        sli = self.slice(origin, normal)
        if sli.n_points == 0:
            raise RuntimeError("Slice has no points.")

        ax = _axis_to_idx(axis)
        axis_normals = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        nvec = axis_normals[ax]
        ds = sli
        if vmin is not None:
            o = [0.0, 0.0, 0.0]; o[ax] = vmin
            ds = ds.clip(normal=nvec, origin=o, invert=False)
        if vmax is not None:
            o = [0.0, 0.0, 0.0]; o[ax] = vmax
            ds = ds.clip(normal=nvec, origin=o, invert=True)
        return ds

    def available_fields(self):
        """List of available normalized field names."""
        fields = list(self._field_map.keys())
        if 'T' not in fields and 'rho' in fields and 'p' in fields:
            fields.append('T')
        return sorted(fields)

    @property
    def points(self):
        return self.mesh.points


def _axis_to_idx(axis):
    if axis in (0, 'x', 'X'): return 0
    if axis in (1, 'y', 'Y'): return 1
    if axis in (2, 'z', 'Z'): return 2
    raise ValueError("axis must be one of: 0,1,2,'x','y','z'")


# -------------------------------------------------------
# plane_local_coords (preserved from original pyfr_post.py)
# -------------------------------------------------------

def plane_local_coords(points, origin, normal, ref_dir=(0, 1, 0)):
    """Project 3D points onto a plane, returning (xi, eta) local coordinates."""
    pts    = np.asarray(points, float)
    origin = np.asarray(origin, float)
    n      = np.asarray(normal, float)
    n      = n / np.linalg.norm(n)

    ref = np.asarray(ref_dir, float)
    ref = ref - np.dot(ref, n) * n

    if np.linalg.norm(ref) < 1e-12:
        raise ValueError("reference direction must differ from normal direction")

    t1 = ref / np.linalg.norm(ref)
    t2 = np.cross(n, t1)

    r   = pts - origin[None, :]
    xi  = r @ t1
    eta = r @ t2
    return xi, eta


# -------------------------------------------------------
# DNS / Reference data loaders
# -------------------------------------------------------

def load_schlatter_dns(re_theta=1410, base_dir=None):
    """Load Schlatter & Orlu (2010) DNS profile.

    Returns dict: y_delta, y_plus, u_plus, urms_plus, vrms_plus,
                  wrms_plus, uv_plus, prms_plus, ...
    """
    if base_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(here, '..', '..', 'reference', 'validation_data', 'dns')

    csv_path = os.path.join(base_dir, f'schlatter_re_theta_{re_theta}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"DNS data not found: {csv_path}")

    import csv
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        _header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader])

    col_names = [
        'y_delta', 'y_plus', 'u_plus', 'urms_plus', 'vrms_plus',
        'wrms_plus', 'uv_plus', 'prms_plus', 'pu_plus', 'pv_plus',
        'Su', 'Fu', 'dUplus_dyplus', 'V_plus',
        'omxrms_plus', 'omyrms_plus', 'omzrms_plus',
    ]
    result = {}
    for i, name in enumerate(col_names):
        if i < data.shape[1]:
            result[name] = data[:, i]
    return result
