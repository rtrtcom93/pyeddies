"""Unit tests for pyeddies.post.interp — Lagrange interpolation engine.

Covers CV1 (basis functions), CV2 (element mapping), CV3 (profile interpolation).
No VTU or PyVista dependency — pure numpy.
"""
import numpy as np
import pytest

from pyeddies.post.interp import (
    lagrange_basis_and_deriv, lagrange_interp_1d,
    build_element_map, detect_elements_from_nodes,
    interpolate_profile, wall_gradient, generate_wall_clustered_y,
)

# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

N = 5
XI_NODES = np.linspace(-1, 1, N)

H_WALL = 0.1824e-3
GROWTH_RATE = 1.15
N_BL_LAYERS = 8


def _make_y_sorted():
    """Generate bin-averaged y-nodes for 8 BL elements."""
    y_all = []
    for i in range(N_BL_LAYERS):
        h_i = H_WALL * GROWTH_RATE ** i
        y_lo = sum(H_WALL * GROWTH_RATE ** j for j in range(i))
        y_all.extend(np.linspace(y_lo, y_lo + h_i, N).tolist())
    return np.array(sorted(set(np.round(y_all, 15))))


# -------------------------------------------------------
# CV1: Lagrange basis functions
# -------------------------------------------------------

class TestCV1:
    def test_identity_at_nodes(self):
        """CV1-1: L(xi_nodes) = I."""
        L, _ = lagrange_basis_and_deriv(XI_NODES, XI_NODES.copy())
        assert np.allclose(L, np.eye(N), atol=1e-14)

    def test_quartic_interpolation(self):
        """CV1-2: 4th-degree polynomial is exact."""
        f = XI_NODES ** 4
        xi_q = np.array([0.25, -0.75, 0.1])
        L, _ = lagrange_basis_and_deriv(XI_NODES, xi_q)
        assert np.allclose(L @ f, xi_q ** 4, atol=1e-12)

    def test_quartic_derivative(self):
        """CV1-3: Derivative of x^4 = 4x^3."""
        f = XI_NODES ** 4
        xi_q = np.array([0.25, -0.75, 0.1])
        _, dL = lagrange_basis_and_deriv(XI_NODES, xi_q)
        assert np.allclose(dL @ f, 4 * xi_q ** 3, atol=1e-11)

    def test_fifth_order_has_error(self):
        """CV1-4: 5th-degree polynomial should NOT be exact."""
        f = XI_NODES ** 5
        xi_q = np.array([0.25, -0.75, 0.1])
        L, _ = lagrange_basis_and_deriv(XI_NODES, xi_q)
        error = np.max(np.abs(L @ f - xi_q ** 5))
        assert error > 1e-4

    def test_linear_derivative_at_node(self):
        """Derivative of f=xi at xi=-1 should be 1."""
        _, dL = lagrange_basis_and_deriv(XI_NODES, np.array([-1.0]))
        df = (dL @ XI_NODES).item()
        assert abs(df - 1.0) < 1e-13

    def test_lagrange_interp_1d_physical(self):
        """lagrange_interp_1d in physical y-space for quartic."""
        y_n = np.linspace(0, 1e-3, N)
        f_n = (y_n * 1e3) ** 4
        y_q = np.array([0.125e-3, 0.625e-3])
        f_q, df_q = lagrange_interp_1d(y_n, f_n, y_q)
        assert np.allclose(f_q, (y_q * 1e3) ** 4, atol=1e-10)
        assert np.allclose(df_q, 4 * (y_q * 1e3) ** 3 * 1e3, atol=1e-8)


# -------------------------------------------------------
# CV2: Element mapping
# -------------------------------------------------------

class TestCV2:
    def setup_method(self):
        self.y_sorted = _make_y_sorted()
        self.elements = build_element_map(
            self.y_sorted, H_WALL, GROWTH_RATE, N_BL_LAYERS)

    def test_element_count(self):
        """CV2-1: Number of BL elements."""
        assert len(self.elements) == N_BL_LAYERS

    def test_nodes_per_element(self):
        """CV2-2: Each element has 5 nodes."""
        for i, elem in enumerate(self.elements):
            assert len(elem['node_indices']) == 5, f"elem {i}"

    def test_equispacing(self):
        """CV2-3: Nodes within each element are equispaced."""
        for i, elem in enumerate(self.elements):
            dy = np.diff(elem['y_nodes'])
            rel_var = np.std(dy) / np.mean(dy)
            assert rel_var < 0.01, f"elem {i}: rel_var={rel_var}"

    def test_boundary_continuity(self):
        """CV2-4: Element boundaries are continuous."""
        for i in range(len(self.elements) - 1):
            gap = abs(self.elements[i + 1]['y_lo'] - self.elements[i]['y_hi'])
            assert gap < 1e-10, f"gap between elem {i} and {i+1}"

    def test_first_element_height(self):
        """CV2-5: First element height matches h_wall."""
        h_first = self.elements[0]['y_hi'] - self.elements[0]['y_lo']
        assert abs(h_first - H_WALL) / H_WALL < 0.05

    def test_auto_detect_matches(self):
        """detect_elements_from_nodes matches build_element_map."""
        auto = detect_elements_from_nodes(self.y_sorted, n_nodes_per_elem=5)
        assert len(auto) == len(self.elements)
        for i in range(len(auto)):
            assert np.allclose(auto[i]['y_nodes'], self.elements[i]['y_nodes'],
                               atol=1e-12)


# -------------------------------------------------------
# CV3: Profile interpolation
# -------------------------------------------------------

class TestCV3:
    def setup_method(self):
        self.y_sorted = _make_y_sorted()
        self.elements = build_element_map(
            self.y_sorted, H_WALL, GROWTH_RATE, N_BL_LAYERS)
        # Linear profile u = a*y
        self.a = 1e6
        self.u_lin = self.a * self.y_sorted

    def test_node_recovery(self):
        """CV3-1: Interpolation at nodes recovers original values."""
        for elem in self.elements:
            f_n = self.u_lin[elem['node_indices']]
            f_q, _ = interpolate_profile(
                self.y_sorted, self.u_lin, [elem], elem['y_nodes'])
            assert np.allclose(f_q, f_n, rtol=1e-12, atol=1e-12)

    def test_boundary_continuity(self):
        """CV3-3: Left/right interpolation at element boundaries match."""
        for i in range(len(self.elements) - 1):
            y_bnd = np.array([self.elements[i]['y_hi']])
            fl, _ = interpolate_profile(
                self.y_sorted, self.u_lin, [self.elements[i]], y_bnd)
            fr, _ = interpolate_profile(
                self.y_sorted, self.u_lin, [self.elements[i + 1]], y_bnd)
            assert np.allclose(fl, fr, rtol=1e-12)

    def test_wall_gradient_linear(self):
        """CV3-4: Wall gradient of linear profile is exact."""
        dudy = wall_gradient(self.elements, self.u_lin)
        assert abs(dudy - self.a) / self.a < 1e-10

    def test_wall_gradient_sign(self):
        """Wall gradient should be positive for increasing velocity."""
        dudy = wall_gradient(self.elements, self.u_lin)
        assert dudy > 0

    def test_viscous_sublayer_points(self):
        """CV3-5: Enough points in viscous sublayer (n_sub=50)."""
        y_query = generate_wall_clustered_y(self.elements, n_sub=50)
        u_tau = 6.45
        nu_w = 2.80237e-5 / 0.52612
        n_visc = np.sum(y_query < 5.0 * nu_w / u_tau)
        assert n_visc >= 10

    def test_dict_interface(self):
        """interpolate_profile works with dict input."""
        fields = {'u': self.u_lin, 'rho': np.ones_like(self.u_lin)}
        y_q = np.array([self.y_sorted[2]])
        f_dict, df_dict = interpolate_profile(
            self.y_sorted, fields, self.elements, y_q)
        assert 'u' in f_dict and 'rho' in f_dict
        assert np.allclose(f_dict['u'], self.a * y_q, rtol=1e-10)

    def test_generate_wall_clustered_y_coverage(self):
        """Generated y covers entire BL domain."""
        y_q = generate_wall_clustered_y(self.elements, n_sub=20)
        assert y_q[0] == pytest.approx(0.0, abs=1e-15)
        assert y_q[-1] == pytest.approx(self.elements[-1]['y_hi'], rel=1e-12)
        # Monotonically increasing
        assert np.all(np.diff(y_q) > 0)
