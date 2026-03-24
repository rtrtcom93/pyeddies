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
    """VTU file wrapper with automatic snapshot/tavg detection.

    Parameters
    ----------
    vtu_path : str
        VTU file path (tavg or instantaneous)
    params : str or FlowParams or dict, optional
        - str → params.yaml 경로 → FlowParams.from_yaml() 자동
        - FlowParams → 직접 사용
        - dict → 기존 호환
        - None → 시각화만 가능, 물리 분석은 비활성
    """

    def __init__(self, vtu_path, params=None):
        if pv is None:
            raise ImportError("pyvista is required for FlowField. Install: pip install pyvista")
        if not os.path.exists(vtu_path):
            raise FileNotFoundError(f"VTU not found: {vtu_path}")
        self.path = vtu_path
        self.mesh = pv.read(vtu_path)
        self.mode = self._detect_mode()
        self._field_map = self._build_field_map()

        # params 연결 (optional)
        self.params = None
        if params is not None:
            if isinstance(params, str):
                from pyeddies.params import FlowParams
                self.params = FlowParams.from_yaml(params)
            else:
                self.params = params

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

    def slice(self, normal_or_origin='z', origin_or_normal=None,
              x_over_D=None):
        """Plane slice with flexible API.

        Supports two calling conventions:
        - New: slice('z'), slice('z', 0.015), slice('x', x_over_D=5)
        - Legacy: slice(origin_tuple, normal_tuple) — auto-detected

        Parameters
        ----------
        normal_or_origin : str or tuple/list
            str ('x','y','z') → axis-aligned slice
            tuple/list of len 3 → could be origin (legacy) or normal vector
        origin_or_normal : float, tuple/list, or None
            str mode: float → coordinate along axis, tuple → [ox,oy,oz]
            legacy mode: normal vector
        x_over_D : float, optional
            Slice at x = x_over_D * D (requires params, normal='x')

        Returns
        -------
        SliceResult or pyvista.PolyData
            SliceResult when using new API; raw PolyData for legacy calls
        """
        import numpy as np

        # --- Legacy detection: slice((ox,oy,oz), (nx,ny,nz)) ---
        if (not isinstance(normal_or_origin, str)
                and origin_or_normal is not None
                and not isinstance(origin_or_normal, (int, float))):
            # Legacy call: slice(origin, normal)
            return self.mesh.slice(normal=origin_or_normal,
                                   origin=normal_or_origin)

        # --- New API ---
        normal = normal_or_origin
        origin = origin_or_normal

        if isinstance(normal, str):
            axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
            n = axis_map[normal]
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[normal]

            if x_over_D is not None and normal == 'x':
                self._require_params('slice with x_over_D')
                fp = self._params_dict()
                o = [0, 0, 0]
                o[axis_idx] = x_over_D * fp['D']
                origin_pt = o
            elif origin is None:
                bounds = self.mesh.bounds
                mid = (bounds[axis_idx * 2] + bounds[axis_idx * 2 + 1]) / 2
                o = [0, 0, 0]
                o[axis_idx] = mid
                origin_pt = o
            elif isinstance(origin, (int, float)):
                o = [0, 0, 0]
                o[axis_idx] = origin
                origin_pt = o
            else:
                origin_pt = list(origin)
        else:
            n = list(normal)
            origin_pt = list(origin) if origin is not None else self.mesh.center

        sliced = self.mesh.slice(normal=n, origin=origin_pt)
        return SliceResult(sliced, self.params)

    def slice_clip(self, origin, normal, axis='y', vmin=None, vmax=None):
        """Slice + clip to a window along an axis.

        Preserved from original PyFRStats._slice_clip_window.
        """
        sli = self.mesh.slice(normal=normal, origin=origin)
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

    def box(self, x=None, y=None, z=None):
        """3D region clipping. No params required.

        Parameters
        ----------
        x, y, z : list of [min, max], optional
            None → full range along that axis

        Returns
        -------
        FlowField — clipped (chaining: ff.box(...).slice('z'))
        """
        bounds = list(self.mesh.bounds)
        if x is not None:
            bounds[0], bounds[1] = x
        if y is not None:
            bounds[2], bounds[3] = y
        if z is not None:
            bounds[4], bounds[5] = z

        clipped = self.mesh.clip_box(bounds, invert=False)
        ff_new = FlowField.__new__(FlowField)
        ff_new.path = self.path
        ff_new.mesh = clipped
        ff_new.mode = self.mode
        ff_new._field_map = self._field_map
        ff_new.params = self.params
        return ff_new

    def probe(self, points):
        """Extract values at specific coordinates.

        Parameters
        ----------
        points : array-like, shape (N, 3)

        Returns
        -------
        dict: {variable_name: array(N)}
        """
        import numpy as np
        pts = np.asarray(points)
        result = self.mesh.probe(pts)
        return {name: result[name] for name in result.array_names}

    def _infer_params_from_vtu(self):
        """Infer flow parameters from VTU freestream data.

        Fallback when no params.yaml provided.
        utau, d99 etc. cannot be inferred → None.
        """
        import warnings
        import numpy as np
        from pyeddies.material.property import get_air_nasa9, mu_sutherland

        R_gas = 287.003
        y = self.mesh.points[:, 1]
        y_top = y.max() * 0.9
        freestream = y > y_top

        # Detect variable names (tavg vs instantaneous)
        names = set(self.mesh.array_names)
        if 'avg-p' in names:
            p_arr, rho_arr, u_arr = 'avg-p', 'avg-rho', 'avg-u'
        elif 'Pressure' in names:
            p_arr, rho_arr, u_arr = 'Pressure', 'Density', 'Velocity'
        else:
            return None

        try:
            p_data = self.mesh[p_arr][freestream]
            rho_data = self.mesh[rho_arr][freestream]
            if u_arr == 'Velocity':
                u_data = self.mesh[u_arr][freestream][:, 0]
            else:
                u_data = self.mesh[u_arr][freestream]
        except (KeyError, IndexError):
            return None

        pe = float(np.mean(p_data))
        rhoe = float(np.mean(rho_data))
        Te = pe / (rhoe * R_gas)
        ue = float(np.mean(u_data))

        air = get_air_nasa9()
        gamma = float(air.gamma(Te))

        warnings.warn(
            f"No params provided — inferred from VTU freestream: "
            f"pe={pe:.0f}, Te={Te:.1f}K, ue={ue:.1f}m/s",
            UserWarning,
        )

        return {
            'pe': pe, 'Te': Te, 'rhoe': rhoe, 'rho_e': rhoe,
            'ue': ue, 'gamma': gamma, 'R_gas': R_gas,
            'mu_e': mu_sutherland(Te),
            '_inferred': True,
        }

    def available_fields(self):
        """List of available normalized field names."""
        fields = list(self._field_map.keys())
        if 'T' not in fields and 'rho' in fields and 'p' in fields:
            fields.append('T')
        return sorted(fields)

    @property
    def points(self):
        return self.mesh.points

    # --- Convenience methods requiring params ---

    def _require_params(self, method_name):
        """Raise if params not set."""
        if self.params is None:
            raise ValueError(
                f"{method_name}() requires flow params. "
                f"Use: FlowField('file.vtu', params='params.yaml')"
            )

    def _params_dict(self):
        """Return params as dict (handles FlowParams, dict, etc.)."""
        if hasattr(self.params, 'to_dict'):
            return self.params.to_dict()
        return self.params

    def sweep(self, x_stations, **kwargs):
        """Create StreamwiseSweep (CTBL).

        Returns
        -------
        StreamwiseSweep
        """
        self._require_params('sweep')
        from .sweep import StreamwiseSweep
        return StreamwiseSweep(self, x_stations, flow_params=self._params_dict(), **kwargs)

    def fc_sweep(self, x_stations, **kwargs):
        """Create FilmCoolingSweep (FC).

        Returns
        -------
        FilmCoolingSweep
        """
        self._require_params('fc_sweep')
        from .fc import FilmCoolingSweep
        return FilmCoolingSweep(self, x_stations, flow_params=self._params_dict(), **kwargs)

    def snapshot_analysis(self, **kwargs):
        """Create SnapshotAnalyzer (instantaneous).

        Returns
        -------
        SnapshotAnalyzer
        """
        from .snapshot import SnapshotAnalyzer
        fp = self._params_dict() if self.params is not None else {}
        return SnapshotAnalyzer(self, flow_params=fp, **kwargs)


def _axis_to_idx(axis):
    if axis in (0, 'x', 'X'): return 0
    if axis in (1, 'y', 'Y'): return 1
    if axis in (2, 'z', 'Z'): return 2
    raise ValueError("axis must be one of: 0,1,2,'x','y','z'")


class SliceResult:
    """Wrapper for slice results — visualization + data access.

    Examples
    --------
    s = ff.slice('z')
    s.plot('Temperature')
    fig, ax = s.plot('Temperature')
    ax.set_title('My title')
    df = s.to_dataframe()
    """

    def __init__(self, mesh_slice, params=None):
        self.mesh = mesh_slice
        self.params = params

    @property
    def data(self):
        """Raw PyVista mesh."""
        return self.mesh

    @property
    def n_points(self):
        return self.mesh.n_points

    @property
    def points(self):
        return self.mesh.points

    def __getitem__(self, key):
        """Access array data by name."""
        return self.mesh[key]

    @property
    def array_names(self):
        return self.mesh.array_names

    def plot(self, variable, **kwargs):
        """2D contour plot.

        Parameters
        ----------
        variable : str
        **kwargs : cmap, clim, title, figsize, etc.

        Returns
        -------
        fig, ax — matplotlib figure and axes
        """
        from pyeddies.viz.contours import plot_slice_contour
        p = None
        if self.params is not None:
            p = self.params.to_dict() if hasattr(self.params, 'to_dict') else self.params
        return plot_slice_contour(self.mesh, variable, params=p, **kwargs)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        coords = self.mesh.points
        data = {'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2]}
        for name in self.mesh.array_names:
            arr = self.mesh[name]
            if arr.ndim == 1:
                data[name] = arr
        return pd.DataFrame(data)


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
