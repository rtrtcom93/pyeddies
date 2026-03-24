"""Mesh grading utilities: auto_grading + volume_ratio_checker.

auto_grading:  1D node spacing with automatic N and growth rate determination.
volume_ratio_checker: 3D hex mesh adjacent-cell volume ratio validation.
"""
import math
import numpy as np


# ================================================================
# 1. auto_grading
# ================================================================
def auto_grading(L_total, h_start, h_end, max_ratio=2.0):
    """Generate 1D graded node distribution [0, L_total].

    Parameters
    ----------
    L_total   : total length of the interval
    h_start   : first cell size at x=0
    h_end     : last cell size at x=L_total (also acts as cap)
    max_ratio : maximum adjacent cell size ratio (default 2.0)

    Returns
    -------
    nodes : ndarray, shape (N+1,), node coordinates from 0 to L_total

    Algorithm
    ---------
    1. r = max_ratio  (start from maximum allowed growth)
    2. Generate cells: h_start → h_start*r → h_start*r² → ... capped at h_end
    3. Continue at h_end until sum >= L_total → determines N
    4. Bisect r in [1, max_ratio] so sum(cells) == L_total exactly
    5. Last cell absorbs tiny rounding remainder
    """
    assert L_total > 0 and h_start > 0 and h_end > 0 and max_ratio > 1.0

    # Reverse case: h_start > h_end → solve mirrored, flip
    if h_start > h_end * 1.01:
        nodes = auto_grading(L_total, h_end, h_start, max_ratio)
        return L_total - nodes[::-1]

    # Nearly uniform case
    if h_end <= h_start * 1.01:
        N = max(1, round(L_total / h_start))
        return np.linspace(0, L_total, N + 1)

    # --- h_start < h_end: grow from h_start, cap at h_end ---

    def count_and_sum(r):
        """Count cells and total length at growth rate r."""
        cells = []
        h = h_start
        total = 0.0
        while total < L_total - 1e-15:
            c = min(h, h_end)
            cells.append(c)
            total += c
            h *= r
        return len(cells), total

    def cell_sum_N(r, N):
        """Sum of exactly N cells at growth rate r, capped at h_end."""
        total = 0.0
        h = h_start
        for _ in range(N):
            total += min(h, h_end)
            h *= r
        return total

    # Step 1: determine N at max growth rate
    N, _ = count_and_sum(max_ratio)

    # Step 2: ensure bounds bracket L_total
    r_lo, r_hi = 1.0 + 1e-12, max_ratio

    # Reduce N if even uniform spacing (r≈1) overshoots
    while N > 1 and cell_sum_N(r_lo, N) > L_total + 1e-10:
        N -= 1

    # Increase N if max growth undershoots
    while cell_sum_N(r_hi, N) < L_total - 1e-10:
        N += 1

    s_lo = cell_sum_N(r_lo, N)
    s_hi = cell_sum_N(r_hi, N)

    # Step 3: bisection on r
    if s_lo >= L_total:
        r = 1.0 + 1e-12  # uniform
    else:
        for _ in range(200):
            r_mid = (r_lo + r_hi) / 2
            s_mid = cell_sum_N(r_mid, N)
            if s_mid < L_total - 1e-15:
                r_lo = r_mid
            else:
                r_hi = r_mid
            if r_hi - r_lo < 1e-14:
                break
        r = (r_lo + r_hi) / 2

    # Step 4: build cells
    cells = np.empty(N)
    h = h_start
    for i in range(N):
        cells[i] = min(h, h_end)
        h *= r

    # Absorb rounding remainder in last cell
    cells[-1] += L_total - cells.sum()

    # Build node array
    nodes = np.zeros(N + 1)
    np.cumsum(cells, out=nodes[1:])
    nodes[-1] = L_total

    # Cross-validation
    d = np.diff(nodes)
    assert np.all(d > -1e-15), f"Negative cell: min={d.min():.3e}"
    if len(d) > 1:
        ratios = d[1:] / d[:-1]
        max_fwd = ratios.max()
        max_bwd = (1.0 / ratios).max()
        tol = 0.05  # 5% tolerance for last-cell absorption
        assert max_fwd <= max_ratio + tol, \
            f"Forward ratio {max_fwd:.4f} > {max_ratio} + tol"
        assert max_bwd <= max_ratio + tol, \
            f"Backward ratio {max_bwd:.4f} > {max_ratio} + tol"

    return nodes


def auto_grading_double(L_total, h_start, h_end, h_max, max_ratio=2.0):
    """Double-sided grading: fine at both ends, coarse (≤ h_max) in middle.

    Parameters
    ----------
    L_total   : total length
    h_start   : first cell at x=0
    h_end     : last cell at x=L_total
    h_max     : maximum cell size in the middle
    max_ratio : max adjacent cell ratio

    Returns
    -------
    nodes : ndarray, shape (N+1,)
    """
    assert L_total > 0 and h_start > 0 and h_end > 0 and h_max > 0

    # Generate left half: h_start → h_max
    # Generate right half: h_end → h_max, reversed
    # Fill middle with h_max cells if needed

    def grow_cells(h0, h_cap, r):
        """Grow from h0 with rate r until reaching h_cap."""
        cells = []
        h = h0
        while h < h_cap * (1 - 1e-10):
            cells.append(h)
            h *= r
        return cells

    def total_for_r(r):
        """Total length of growing regions + middle fill at rate r."""
        left = grow_cells(h_start, min(h_max, h_start * r**50), r)
        right = grow_cells(h_end, min(h_max, h_end * r**50), r)
        s_grow = sum(left) + sum(right)
        remaining = L_total - s_grow
        if remaining <= 0:
            return s_grow, left, right, 0
        N_mid = max(0, round(remaining / h_max))
        return s_grow + N_mid * h_max, left, right, N_mid

    # Bisect on growth rate r
    r_lo, r_hi = 1.0 + 1e-12, max_ratio

    # Check if we even need grading
    if h_start >= h_max * 0.99 and h_end >= h_max * 0.99:
        N = max(1, round(L_total / h_max))
        return np.linspace(0, L_total, N + 1)

    # Find r that makes the total fit
    best_r = max_ratio
    for _ in range(200):
        r_mid = (r_lo + r_hi) / 2
        s, left, right, N_mid = total_for_r(r_mid)
        if s < L_total - 1e-10:
            r_hi = r_mid  # need slower growth → more cells
        else:
            r_lo = r_mid
        if r_hi - r_lo < 1e-12:
            break
        best_r = r_mid

    # Rebuild with best_r
    r = (r_lo + r_hi) / 2
    left_cells = grow_cells(h_start, min(h_max, h_start * r**50), r)
    right_cells = grow_cells(h_end, min(h_max, h_end * r**50), r)
    s_grow = sum(left_cells) + sum(right_cells)
    remaining = L_total - s_grow

    if remaining > 0:
        N_mid = max(1, round(remaining / h_max))
        mid_cell = remaining / N_mid
        mid_cells = [mid_cell] * N_mid
    else:
        mid_cells = []

    all_cells = left_cells + mid_cells + list(reversed(right_cells))

    # Absorb rounding
    total = sum(all_cells)
    if abs(total - L_total) > 1e-15 and len(all_cells) > 0:
        all_cells[-1] += L_total - total

    nodes = np.zeros(len(all_cells) + 1)
    for i, c in enumerate(all_cells):
        nodes[i + 1] = nodes[i] + c
    nodes[-1] = L_total

    # Cross-validation
    d = np.diff(nodes)
    assert np.all(d > -1e-15), f"Negative cell: min={min(d):.3e}"
    if len(d) > 1:
        ratios = d[1:] / d[:-1]
        tol = 0.10  # slightly more tolerance for double-sided join
        assert ratios.max() <= max_ratio + tol, \
            f"Forward ratio {ratios.max():.4f} > {max_ratio}"
        assert (1.0 / ratios).max() <= max_ratio + tol, \
            f"Backward ratio {(1.0/ratios).max():.4f} > {max_ratio}"

    return nodes


# ================================================================
# 2. volume_ratio_checker
# ================================================================
def volume_ratio_checker(nodes_coords, elements, max_ratio=2.0):
    """Check adjacent hex element volume ratios.

    Parameters
    ----------
    nodes_coords : ndarray, shape (N_nodes, 3)
    elements     : ndarray, shape (N_elem, 8), hex connectivity (0-based)
    max_ratio    : maximum allowed volume ratio

    Returns
    -------
    dict with keys: max_ratio, mean_ratio, n_violations, violations,
                    histogram, pass
    """
    n_elem = len(elements)

    # Compute volumes (5-tet decomposition)
    def _tet_vol(a, b, c, d):
        return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

    volumes = np.zeros(n_elem)
    for i in range(n_elem):
        c = nodes_coords[elements[i]]
        volumes[i] = (_tet_vol(c[0], c[1], c[3], c[4])
                      + _tet_vol(c[1], c[2], c[3], c[6])
                      + _tet_vol(c[1], c[4], c[5], c[6])
                      + _tet_vol(c[3], c[4], c[6], c[7])
                      + _tet_vol(c[1], c[3], c[4], c[6]))

    # Build face → element adjacency
    hex_faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (0, 3, 7, 4), (1, 2, 6, 5),
    ]
    face_to_elem = {}
    for ei in range(n_elem):
        h = elements[ei]
        for fi in hex_faces:
            key = tuple(sorted(h[k] for k in fi))
            if key in face_to_elem:
                face_to_elem[key].append(ei)
            else:
                face_to_elem[key] = [ei]

    # Compute ratios
    ratios_list = []
    violations = []
    for face, elems in face_to_elem.items():
        if len(elems) == 2:
            e0, e1 = elems
            v0, v1 = volumes[e0], volumes[e1]
            if min(v0, v1) > 0:
                r = max(v0, v1) / min(v0, v1)
            else:
                r = float('inf')
            ratios_list.append(r)
            if r > max_ratio:
                violations.append((e0, e1, r))

    r_arr = np.array(ratios_list) if ratios_list else np.array([1.0])

    # Histogram
    bins = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, float('inf')]
    counts, bin_edges = np.histogram(r_arr, bins=bins)

    return {
        'max_ratio': float(r_arr.max()),
        'mean_ratio': float(r_arr.mean()),
        'n_violations': len(violations),
        'n_pairs': len(ratios_list),
        'violations': violations,
        'histogram': (counts, bins),
        'pass': len(violations) == 0,
    }


def print_volume_ratio_report(result):
    """Pretty-print volume_ratio_checker result."""
    print(f"  Volume Ratio Check:")
    print(f"    Max ratio:   {result['max_ratio']:.4f}")
    print(f"    Mean ratio:  {result['mean_ratio']:.4f}")
    print(f"    Violations:  {result['n_violations']} / {result['n_pairs']} pairs")

    counts, bins = result['histogram']
    print(f"    Histogram:")
    for i in range(len(counts)):
        lo = bins[i]
        hi = bins[i + 1]
        hi_s = f"{hi:.1f}" if hi < 100 else "inf"
        pct = 100 * counts[i] / result['n_pairs'] if result['n_pairs'] > 0 else 0
        print(f"      [{lo:.1f}, {hi_s}): {counts[i]:>6d} ({pct:5.1f}%)")

    status = "PASS" if result['pass'] else "FAIL"
    print(f"    {status}")
