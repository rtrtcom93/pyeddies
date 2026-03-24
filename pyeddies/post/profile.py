"""profile.py — Wall-normal profile extraction, binning, spanwise averaging.

Preserved logic from original SliceStats._get_avg (ndigits binning).
"""
import os

import numpy as np

from .core import FlowField


class WallNormalProfile:
    """Wall-normal profile at a single streamwise station.

    Stores sorted y-coordinate and field arrays.
    Access fields via attribute: profile.u, profile.rho, etc.
    """

    def __init__(self, y, fields_dict):
        """
        Parameters
        ----------
        y : np.ndarray
            Wall-normal coordinates, sorted ascending.
        fields_dict : dict
            {field_name: np.ndarray} all same length as y.
        """
        self.y = np.asarray(y, dtype=float)
        self._fields = dict(fields_dict)

    def __getattr__(self, name):
        if name.startswith('_') or name == 'y':
            raise AttributeError(name)
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"Field '{name}' not in profile. "
                             f"Available: {list(self._fields.keys())}")

    def has_field(self, name):
        return name in self._fields

    def field_names(self):
        return sorted(self._fields.keys())

    def to_csv(self, path):
        """Save profile to CSV."""
        import csv
        cols = ['y'] + sorted(self._fields.keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for i in range(len(self.y)):
                row = [self.y[i]] + [self._fields[c][i] for c in cols[1:]]
                writer.writerow(row)

    @classmethod
    def from_csv(cls, path):
        """Restore profile from CSV (no VTU reprocessing needed)."""
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        with open(path, encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        y = data[:, 0]
        fields = {name: data[:, i] for i, name in enumerate(header) if i > 0}
        return cls(y, fields)

    def to_npz(self, path):
        """Save profile to npz."""
        np.savez(path, y=self.y, **self._fields)

    @classmethod
    def from_npz(cls, path):
        """Restore from npz."""
        d = np.load(path)
        y = d['y']
        fields = {k: d[k] for k in d.files if k != 'y'}
        return cls(y, fields)


# -------------------------------------------------------
# Binning + spanwise averaging (preserved from SliceStats._get_avg)
# -------------------------------------------------------

def _bin_average(coords, var, ndigits=9):
    """Average var in coordinate bins.

    Preserved from original SliceStats._get_avg logic.
    ndigits=9 prevents grid aliasing noise near walls.
    """
    e_bin = np.round(coords, ndigits)
    e_unique = np.unique(e_bin)

    out_coords = []
    out_stats = []

    for eb in e_unique:
        mask = (e_bin == eb)
        out_coords.append(coords[mask].mean())
        out_stats.append(var[mask].mean())

    out_coords = np.array(out_coords)
    out_stats = np.array(out_stats)
    idx = np.argsort(out_coords)
    return out_coords[idx], out_stats[idx]


def extract_wall_normal_profile(flow_field, x_target, z_mid=None,
                                 wall_normal='y', ndigits=9,
                                 R_gas=287.0):
    """Extract spanwise-averaged wall-normal profile from FlowField.

    Uses an x-normal slice at x_target to get a y-z cross-section,
    then bin-averages along y (wall-normal). This naturally performs
    spanwise averaging over all z-points on that plane.

    Parameters
    ----------
    flow_field : FlowField
        Loaded VTU data.
    x_target : float
        Streamwise x-position [m] for the slice.
    z_mid : float or None
        If given, filter to z ≈ z_mid before bin-averaging (single-plane).
        If None, spanwise-average over all z points.
    wall_normal : str
        'y' (default) — wall-normal direction.
    ndigits : int
        Coordinate binning precision (preserved from original).
    R_gas : float
        Gas constant for T = p/(rho*R).

    Returns
    -------
    WallNormalProfile
    """
    # x-normal slice at x_target → y-z cross-section
    origin = (x_target, 0.0, 0.0)
    normal = (1.0, 0.0, 0.0)

    sli = flow_field.slice(origin, normal)
    if sli.n_points == 0:
        raise RuntimeError(f"Slice at x={x_target} has no points.")

    # Wall-normal coordinate from slice points
    y_raw = sli.points[:, 1]

    # Optional: filter to z ≈ z_mid (single spanwise plane)
    if z_mid is not None:
        z_coords = sli.points[:, 2]
        z_binned = np.round(z_coords, ndigits)
        z_unique = np.unique(z_binned)
        z_nearest = z_unique[np.argmin(np.abs(z_unique - z_mid))]
        z_mask = (z_binned == z_nearest)
        y_raw = y_raw[z_mask]
    else:
        z_mask = None

    avail = flow_field.available_fields()

    # Extract each field from the slice
    raw_fields = {}
    for fname in avail:
        try:
            if fname == 'T':
                p_sli = _get_slice_field(sli, flow_field, 'p')
                rho_sli = _get_slice_field(sli, flow_field, 'rho')
                if p_sli is not None and rho_sli is not None:
                    T_data = p_sli / (rho_sli * R_gas)
                    raw_fields['T'] = T_data[z_mask] if z_mask is not None else T_data
            else:
                data = _get_slice_field(sli, flow_field, fname)
                if data is not None:
                    raw_fields[fname] = data[z_mask] if z_mask is not None else data
        except Exception:
            pass

    if 'u' not in raw_fields or 'rho' not in raw_fields:
        raise RuntimeError("Essential fields (u, rho) not found in slice.")

    # Bin-average all fields along wall-normal direction
    # (averages over z if no z_mid filter, giving spanwise average)
    binned_fields = {}
    y_binned = None
    for fname, data in raw_fields.items():
        y_out, val_out = _bin_average(y_raw, data, ndigits=ndigits)
        if y_binned is None:
            y_binned = y_out
        binned_fields[fname] = val_out

    return WallNormalProfile(y_binned, binned_fields)


def extract_profile_lagrange(flow_field, x_target, mesh_params,
                             y_query=None, n_sub=50, z_mid=None,
                             ndigits=9, R_gas=287.0):
    """Extract wall-normal profile with Lagrange interpolation.

    Same interface as extract_wall_normal_profile, but applies p=4
    Lagrange interpolation within each element for sub-element resolution.

    Parameters
    ----------
    flow_field : FlowField
        Loaded VTU data.
    x_target : float
        Streamwise x-position [m].
    mesh_params : dict
        Must contain: 'h_wall', 'growth_rate', 'n_bl_layers'.
        Optional: 'n_nodes_per_elem' (default 5 for order=4).
    y_query : np.ndarray or None
        Interpolation y-positions. If None, auto-generated via
        generate_wall_clustered_y with n_sub subdivisions per element.
    n_sub : int
        Subdivisions per element (used when y_query is None).
    z_mid : float or None
        If given, filter to z ≈ z_mid (single spanwise plane).
    ndigits : int
        Coordinate binning precision.
    R_gas : float
        Gas constant for T = p/(rho*R).

    Returns
    -------
    WallNormalProfile
        Lagrange-interpolated profile at y_query positions.
    list[dict]
        Element map (for subsequent wall_gradient calls).
    """
    from .interp import (build_element_map, detect_elements_from_nodes,
                         interpolate_profile, generate_wall_clustered_y)

    # Step 1: Extract bin-averaged profile at VTU node positions (existing method)
    prof_nodes = extract_wall_normal_profile(
        flow_field, x_target, z_mid=z_mid, ndigits=ndigits, R_gas=R_gas
    )

    y_sorted = prof_nodes.y
    h_wall = mesh_params['h_wall']
    growth_rate = mesh_params['growth_rate']
    n_bl_layers = mesh_params['n_bl_layers']
    n_nodes_per_elem = mesh_params.get('n_nodes_per_elem', 5)

    # Step 2: Build element map from known mesh structure
    # Fall back to auto-detection if mesh params don't match exactly
    try:
        elements = build_element_map(
            y_sorted, h_wall, growth_rate, n_bl_layers,
            n_nodes_per_elem=n_nodes_per_elem
        )
    except RuntimeError:
        elements = None

    # Auto-detect all elements (robust: uses node spacing patterns, no params needed)
    all_elements = detect_elements_from_nodes(
        y_sorted, n_nodes_per_elem=n_nodes_per_elem)

    if elements is None:
        import warnings
        warnings.warn(
            "build_element_map failed (mesh params mismatch). "
            "Using detect_elements_from_nodes for all elements.",
            stacklevel=2)
        elements = all_elements

    # Step 3: Generate query points if not provided
    # Use all detected elements for full-domain coverage
    if y_query is None:
        y_query = generate_wall_clustered_y(all_elements, n_sub=n_sub)

    # Step 4: Lagrange-interpolate all fields
    fields_dict = {}
    for fname in prof_nodes.field_names():
        fvals = getattr(prof_nodes, fname)
        fields_dict[fname] = fvals

    f_interp, _ = interpolate_profile(
        y_sorted, fields_dict, all_elements, y_query)

    return WallNormalProfile(y_query, f_interp), all_elements, prof_nodes


def _get_slice_field(sli, flow_field, name):
    """Extract a normalized field from a PyVista slice DataSet."""
    fmap = flow_field._field_map
    if name in fmap:
        vtu_name, comp = fmap[name]
        if vtu_name in sli.array_names:
            data = sli.point_data[vtu_name]
            if comp is not None:
                return data[:, comp]
            return data
    return None
