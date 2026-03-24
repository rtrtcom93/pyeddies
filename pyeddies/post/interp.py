"""interp.py — Lagrange interpolation for high-order (p=4) VTU profile extraction.

Provides element-wise p=4 Lagrange interpolation on equispaced VTU nodes,
enabling sub-element resolution and analytical wall gradients.

VTU exported with --eopt=order=4 produces VTK_LAGRANGE_HEXAHEDRON with
5×5×5 equispaced nodes per element (ξ = -1, -0.5, 0, 0.5, 1).
PyVista's sample()/probe() falls back to linear interpolation,
but since VTU node values ARE exact p=4 polynomial values at equispaced
points, we can reconstruct the original solution via Lagrange interpolation.
"""

import numpy as np


# -------------------------------------------------------
# Step 1: Lagrange basis functions + derivatives
# -------------------------------------------------------

def lagrange_basis_and_deriv(xi_nodes: np.ndarray,
                             xi_query: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lagrange basis function values and derivatives.

    Parameters
    ----------
    xi_nodes : shape (N,)
        Reference-space node coordinates (e.g. [-1, -0.5, 0, 0.5, 1] for N=5).
    xi_query : shape (M,)
        Query coordinates in reference space.

    Returns
    -------
    L : shape (M, N)
        L[m, k] = L_k(ξ_m), Lagrange basis value.
    dL : shape (M, N)
        dL[m, k] = L_k'(ξ_m), Lagrange basis derivative.
    """
    xi_nodes = np.asarray(xi_nodes, dtype=float)
    xi_query = np.asarray(xi_query, dtype=float)
    N = len(xi_nodes)
    M = len(xi_query)

    L = np.zeros((M, N))
    dL = np.zeros((M, N))

    for m in range(M):
        xi = xi_query[m]

        # Check if xi coincides with a node (singularity handling)
        diffs = xi - xi_nodes  # shape (N,)
        coincident = np.abs(diffs) < 1e-14
        if np.any(coincident):
            # xi is at node k_hit → L_k_hit = 1, others = 0
            k_hit = np.where(coincident)[0][0]
            L[m, k_hit] = 1.0
            # Derivative at node: L_k'(ξ_k) = Σ_{j≠k} 1/(ξ_k - ξ_j)
            # L_j'(ξ_k) = Π_{i≠j,i≠k} (ξ_k - ξ_i) / Π_{i≠j} (ξ_j - ξ_i)
            for k in range(N):
                if k == k_hit:
                    # dL_k/dξ at ξ = ξ_k
                    s = 0.0
                    for j in range(N):
                        if j != k:
                            s += 1.0 / (xi_nodes[k] - xi_nodes[j])
                    dL[m, k] = s
                else:
                    # dL_k/dξ at ξ = ξ_{k_hit}  (k ≠ k_hit)
                    # L_k'(ξ_{k_hit}) = Π_{j≠k,j≠k_hit}(ξ_{k_hit}-ξ_j) / Π_{j≠k}(ξ_k-ξ_j)
                    num = 1.0
                    den = 1.0
                    for j in range(N):
                        if j != k and j != k_hit:
                            num *= (xi_nodes[k_hit] - xi_nodes[j])
                        if j != k:
                            den *= (xi_nodes[k] - xi_nodes[j])
                    dL[m, k] = num / den
        else:
            # General case: standard Lagrange formula
            for k in range(N):
                # L_k(ξ) = Π_{j≠k} (ξ - ξ_j) / (ξ_k - ξ_j)
                basis_val = 1.0
                for j in range(N):
                    if j != k:
                        basis_val *= diffs[j] / (xi_nodes[k] - xi_nodes[j])
                L[m, k] = basis_val

                # L_k'(ξ) = L_k(ξ) × Σ_{j≠k} 1/(ξ - ξ_j)
                deriv_sum = 0.0
                for j in range(N):
                    if j != k:
                        deriv_sum += 1.0 / diffs[j]
                dL[m, k] = basis_val * deriv_sum

    return L, dL


def lagrange_interp_1d(y_nodes: np.ndarray,
                       f_nodes: np.ndarray,
                       y_query: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange interpolation and derivative in physical y-space.

    Maps y → ξ ∈ [-1, 1], computes Lagrange basis, then evaluates
    f(y_query) and df/dy(y_query) via matrix multiplication.

    Parameters
    ----------
    y_nodes : shape (N,)
        Physical y-coordinates of element nodes (ascending).
    f_nodes : shape (N,) or (N, F)
        Field values at nodes. F = number of fields for batch processing.
    y_query : shape (M,)
        Physical y-coordinates to interpolate at.

    Returns
    -------
    f_interp : shape (M,) or (M, F)
        Interpolated field values.
    df_dy_interp : shape (M,) or (M, F)
        Interpolated df/dy values.
    """
    y_nodes = np.asarray(y_nodes, dtype=float)
    f_nodes = np.asarray(f_nodes, dtype=float)
    y_query = np.asarray(y_query, dtype=float)

    N = len(y_nodes)
    y_lo, y_hi = y_nodes[0], y_nodes[-1]
    h = y_hi - y_lo

    # Map y → ξ ∈ [-1, 1]
    xi_nodes = np.linspace(-1.0, 1.0, N)
    xi_query = 2.0 * (y_query - y_lo) / h - 1.0

    # Basis functions and derivatives in reference space
    L, dL = lagrange_basis_and_deriv(xi_nodes, xi_query)

    # dξ/dy = 2 / h
    dxi_dy = 2.0 / h

    # Interpolation via matrix multiply
    f_interp = L @ f_nodes            # (M, N) @ (N,) → (M,) or (N,F) → (M,F)
    df_dxi = dL @ f_nodes             # derivative in ξ-space
    df_dy_interp = df_dxi * dxi_dy    # chain rule

    return f_interp, df_dy_interp


# -------------------------------------------------------
# Step 2: Element boundary identification + node grouping
# -------------------------------------------------------

def build_element_map(y_sorted: np.ndarray,
                      h_wall: float,
                      growth_rate: float,
                      n_bl_layers: int,
                      n_nodes_per_elem: int = 5,
                      tol_frac: float = 0.1
                      ) -> list[dict]:
    """Identify element boundaries and group nodes using known mesh structure.

    For BL elements with geometric growth, the mesh structure is fully known:
    element i has height h_wall * growth_rate^i.  VTU order=4 gives 5
    equispaced nodes per element along y.

    After bin_average, shared boundary nodes may be merged (giving
    n_bl_layers*4 + 1 unique y-values) or kept separate (n_bl_layers*5).
    Both cases are handled.

    Parameters
    ----------
    y_sorted : shape (P,)
        Ascending y-coordinates from bin-averaged slice.
    h_wall : float
        First element wall-normal height [m].
    growth_rate : float
        Geometric growth rate for BL elements.
    n_bl_layers : int
        Number of BL elements.
    n_nodes_per_elem : int
        Nodes per element in y-direction (5 for VTU order=4).
    tol_frac : float
        Fraction of element height used as matching tolerance.

    Returns
    -------
    elements : list of dict
        Each dict has keys:
        - 'y_lo': float — element bottom y
        - 'y_hi': float — element top y
        - 'node_indices': list[int] — indices into y_sorted
        - 'y_nodes': np.ndarray — shape (n_nodes_per_elem,) node y-coordinates
    """
    y_sorted = np.asarray(y_sorted, dtype=float)

    # Compute expected element edges
    y_edges = [0.0]
    for i in range(n_bl_layers):
        y_edges.append(y_edges[-1] + h_wall * growth_rate ** i)

    elements = []
    for i in range(n_bl_layers):
        y_lo = y_edges[i]
        y_hi = y_edges[i + 1]
        h_i = y_hi - y_lo
        tol = h_i * tol_frac

        # Expected equispaced node positions within this element
        y_expected = np.linspace(y_lo, y_hi, n_nodes_per_elem)

        # Match each expected node to the nearest point in y_sorted
        node_indices = []
        for y_exp in y_expected:
            dists = np.abs(y_sorted - y_exp)
            idx = np.argmin(dists)
            if dists[idx] > tol:
                raise RuntimeError(
                    f"Element {i}: no node near y={y_exp:.6e} "
                    f"(nearest y={y_sorted[idx]:.6e}, dist={dists[idx]:.2e}, tol={tol:.2e})")
            node_indices.append(int(idx))

        # Ensure unique indices (bin_average may map two expected positions
        # to the same y_sorted index at element boundaries)
        unique_indices = list(dict.fromkeys(node_indices))  # preserves order
        if len(unique_indices) < n_nodes_per_elem:
            # Boundary node was shared — try to find the adjacent duplicate
            # and resolve by picking distinct neighbors
            raise RuntimeError(
                f"Element {i}: only {len(unique_indices)} unique nodes found "
                f"(expected {n_nodes_per_elem}). Check bin_average merging.")

        y_nodes = y_sorted[node_indices]
        elements.append({
            'y_lo': y_lo,
            'y_hi': y_hi,
            'node_indices': node_indices,
            'y_nodes': y_nodes,
        })

    return elements


def detect_elements_from_nodes(y_sorted: np.ndarray,
                               n_nodes_per_elem: int = 5
                               ) -> list[dict]:
    """Detect element boundaries from node spacing patterns (no mesh params needed).

    Within an element, the n_nodes_per_elem nodes are equispaced.
    At element boundaries, the spacing pattern changes (either zero gap
    for shared boundary nodes, or a different spacing from the next element).

    Strategy: consecutive groups of n_nodes_per_elem nodes with nearly
    equal internal spacing form one element. Boundary nodes (last of elem i
    = first of elem i+1) are shared.

    Parameters
    ----------
    y_sorted : shape (P,)
        Ascending y-coordinates.
    n_nodes_per_elem : int
        Nodes per element (5 for VTU order=4).

    Returns
    -------
    elements : list of dict
        Same format as build_element_map output.
    """
    y_sorted = np.asarray(y_sorted, dtype=float)
    P = len(y_sorted)

    # Spacing between consecutive nodes
    dy = np.diff(y_sorted)

    # Number of interior intervals per element
    n_int = n_nodes_per_elem - 1  # 4 for order=4

    elements = []
    i = 0  # current start index in y_sorted

    while i + n_int <= P - 1:
        # Try group [i, i+n_int] as one element
        indices = list(range(i, i + n_nodes_per_elem))
        spacings = dy[i:i + n_int]

        # Check equispacing: all intervals nearly equal
        mean_sp = np.mean(spacings)
        if mean_sp < 1e-15:
            i += 1
            continue

        rel_var = np.std(spacings) / mean_sp
        if rel_var > 0.05:
            # Not equispaced — skip this start position
            i += 1
            continue

        y_nodes = y_sorted[indices]
        elements.append({
            'y_lo': float(y_nodes[0]),
            'y_hi': float(y_nodes[-1]),
            'node_indices': indices,
            'y_nodes': y_nodes,
        })

        # Next element starts at the last node of this element (shared boundary)
        i += n_int

    return elements


# -------------------------------------------------------
# Step 3: Profile interpolation, wall gradient, y-generation
# -------------------------------------------------------

def interpolate_profile(y_sorted: np.ndarray,
                        field_values: np.ndarray | dict,
                        elements: list[dict],
                        y_query: np.ndarray
                        ) -> tuple:
    """Interpolate field values at arbitrary y positions using element-wise Lagrange.

    Parameters
    ----------
    y_sorted : shape (P,)
        Original node y-coordinates (ascending).
    field_values : shape (P,) or dict{name: (P,)}
        Field values at nodes.
    elements : list of dict
        From build_element_map or detect_elements_from_nodes.
    y_query : shape (M,)
        Interpolation positions.

    Returns
    -------
    f_interp : shape (M,) or dict{name: (M,)}
    df_dy : shape (M,) or dict{name: (M,)}
    """
    y_query = np.asarray(y_query, dtype=float)
    M = len(y_query)

    # Build element edge array for binary search
    elem_lo = np.array([e['y_lo'] for e in elements])
    elem_hi = np.array([e['y_hi'] for e in elements])

    def _find_element(y_val):
        """Find which element contains y_val (binary search)."""
        for idx in range(len(elements)):
            if elem_lo[idx] - 1e-15 <= y_val <= elem_hi[idx] + 1e-15:
                return idx
        # Extrapolation: use nearest element
        if y_val < elem_lo[0]:
            return 0
        return len(elements) - 1

    def _interp_single_field(fvals):
        """Interpolate a single field array."""
        f_out = np.zeros(M)
        df_out = np.zeros(M)

        for m in range(M):
            eidx = _find_element(y_query[m])
            elem = elements[eidx]
            y_n = elem['y_nodes']
            f_n = fvals[elem['node_indices']]
            fi, dfi = lagrange_interp_1d(y_n, f_n, y_query[m:m+1])
            f_out[m] = fi[0]
            df_out[m] = dfi[0]

        return f_out, df_out

    if isinstance(field_values, dict):
        f_interp = {}
        df_dy = {}
        for name, fvals in field_values.items():
            f_interp[name], df_dy[name] = _interp_single_field(
                np.asarray(fvals, dtype=float))
        return f_interp, df_dy
    else:
        return _interp_single_field(np.asarray(field_values, dtype=float))


def wall_gradient(elements: list[dict],
                  field_values: np.ndarray,
                  y_sorted: np.ndarray | None = None
                  ) -> float:
    """Compute df/dy at wall (y=0) using analytical Lagrange derivative.

    Uses the first element's 5 nodes for a 4th-order polynomial derivative
    at ξ = -1 (wall).  More accurate than the 2-point quadratic fit
    in wall.py (5 points, 4th order vs 2 points, 1st order).

    Parameters
    ----------
    elements : list of dict
        Element map (first element must be the wall element).
    field_values : shape (P,)
        Field values at all y_sorted nodes.
    y_sorted : shape (P,) or None
        Not used (kept for API consistency). Node y-coordinates are
        taken from elements[0]['y_nodes'].

    Returns
    -------
    df_dy_wall : float
        df/dy evaluated at the wall (y = y_lo of first element).
    """
    elem = elements[0]
    y_nodes = elem['y_nodes']
    f_nodes = np.asarray(field_values, dtype=float)[elem['node_indices']]

    N = len(y_nodes)
    xi_nodes = np.linspace(-1.0, 1.0, N)
    xi_wall = np.array([-1.0])

    _, dL = lagrange_basis_and_deriv(xi_nodes, xi_wall)
    df_dxi_wall = (dL @ f_nodes).item()

    h = y_nodes[-1] - y_nodes[0]
    df_dy_wall = df_dxi_wall * 2.0 / h

    return df_dy_wall


def generate_wall_clustered_y(elements: list[dict],
                              n_sub: int = 20
                              ) -> np.ndarray:
    """Generate wall-clustered y-coordinates by subdividing each element.

    Each element is divided into n_sub equally-spaced sub-intervals,
    producing dense sampling near the wall (where elements are thin)
    and coarser sampling far from the wall.

    Parameters
    ----------
    elements : list of dict
        Element map from build_element_map or detect_elements_from_nodes.
    n_sub : int
        Number of sub-intervals per element.

    Returns
    -------
    y_query : shape (n_elements * n_sub + 1,)
        Wall-clustered y-coordinates (ascending, unique).
    """
    y_list = []
    for elem in elements:
        y_lo = elem['y_lo']
        y_hi = elem['y_hi']
        # n_sub intervals → n_sub+1 points, but exclude last (shared with next)
        y_sub = np.linspace(y_lo, y_hi, n_sub + 1)
        y_list.append(y_sub[:-1])

    # Add the last element's upper boundary
    y_list.append(np.array([elements[-1]['y_hi']]))

    y_query = np.concatenate(y_list)
    y_query = np.unique(y_query)
    return y_query
