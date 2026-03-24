"""Instantaneous snapshot analysis — vortex ID, derived fields, slicing."""
import glob
import re
import numpy as np


class SnapshotAnalyzer:
    """Instantaneous VTU 분석기.

    Parameters
    ----------
    ff : FlowField
        Instantaneous snapshot (not tavg)
    flow_params : dict
        Required: gamma, R_gas, rhoe, ue, pe
    """

    def __init__(self, ff, flow_params):
        self.ff = ff
        self.params = flow_params

    # --- Derived fields ---

    def compute_temperature(self):
        """T = p / (rho * R_gas)"""
        rho = self.ff.mesh.point_data['rho']
        p = self.ff.mesh.point_data['p']
        R = self.params['R_gas']
        T = p / (rho * R)
        self.ff.mesh.point_data['T_computed'] = T
        return T

    def compute_mach(self):
        """M = |V| / sqrt(gamma * R_gas * T)"""
        gamma = self.params['gamma']
        R = self.params['R_gas']
        u = self.ff.mesh.point_data['u']
        v = self.ff.mesh.point_data['v']
        w = self.ff.mesh.point_data['w']
        V_mag = np.sqrt(u**2 + v**2 + w**2)

        if 'T_computed' in self.ff.mesh.point_data:
            T = self.ff.mesh.point_data['T_computed']
        else:
            T = self.compute_temperature()

        a = np.sqrt(gamma * R * T)
        M = V_mag / a
        self.ff.mesh.point_data['Mach'] = M
        return M

    def compute_vorticity(self):
        """ω = ∇ × V  (PyVista gradient 기반)"""
        import pyvista as pv
        mesh = self.ff.mesh
        # Stack velocity into a vector
        u = mesh.point_data['u']
        v = mesh.point_data['v']
        w = mesh.point_data['w']
        mesh.point_data['velocity'] = np.column_stack([u, v, w])
        grad = mesh.compute_derivative('velocity')
        # Vorticity components from gradient tensor
        # grad has 9 components: du/dx, du/dy, du/dz, dv/dx, ...
        g = grad.point_data['gradient'].reshape(-1, 3, 3)
        omega_x = g[:, 2, 1] - g[:, 1, 2]  # dw/dy - dv/dz
        omega_y = g[:, 0, 2] - g[:, 2, 0]  # du/dz - dw/dx
        omega_z = g[:, 1, 0] - g[:, 0, 1]  # dv/dx - du/dy
        omega = np.column_stack([omega_x, omega_y, omega_z])
        mesh.point_data['vorticity'] = omega
        return omega

    def compute_q_criterion(self):
        """Q = 0.5*(|Ω|² - |S|²), S=strain, Ω=rotation"""
        mesh = self.ff.mesh
        if 'velocity' not in mesh.point_data:
            u = mesh.point_data['u']
            v = mesh.point_data['v']
            w = mesh.point_data['w']
            mesh.point_data['velocity'] = np.column_stack([u, v, w])

        grad = mesh.compute_derivative('velocity')
        g = grad.point_data['gradient'].reshape(-1, 3, 3)

        S = 0.5 * (g + g.transpose(0, 2, 1))  # strain rate
        Omega = 0.5 * (g - g.transpose(0, 2, 1))  # rotation

        S_norm2 = np.sum(S * S, axis=(1, 2))
        Omega_norm2 = np.sum(Omega * Omega, axis=(1, 2))
        Q = 0.5 * (Omega_norm2 - S_norm2)

        mesh.point_data['Q_criterion'] = Q
        return Q

    def compute_lambda2(self):
        """λ2 criterion (Jeong & Hussain 1995)"""
        mesh = self.ff.mesh
        if 'velocity' not in mesh.point_data:
            u = mesh.point_data['u']
            v = mesh.point_data['v']
            w = mesh.point_data['w']
            mesh.point_data['velocity'] = np.column_stack([u, v, w])

        grad = mesh.compute_derivative('velocity')
        g = grad.point_data['gradient'].reshape(-1, 3, 3)

        S = 0.5 * (g + g.transpose(0, 2, 1))
        Omega = 0.5 * (g - g.transpose(0, 2, 1))

        M = S @ S + Omega @ Omega
        eigvals = np.linalg.eigvalsh(M)  # sorted ascending
        lam2 = eigvals[:, 1]  # second eigenvalue

        mesh.point_data['lambda2'] = lam2
        return lam2

    def compute_eta_instantaneous(self, Taw, Tc):
        """Instantaneous η = (Taw - T) / (Taw - Tc)"""
        if 'T_computed' in self.ff.mesh.point_data:
            T = self.ff.mesh.point_data['T_computed']
        else:
            T = self.compute_temperature()
        eta = (Taw - T) / (Taw - Tc)
        self.ff.mesh.point_data['eta'] = eta
        return eta

    # --- Slicing ---

    def slice_z(self, z_val=None):
        """z = const plane (centerline). z_val=None → Lz/2."""
        if z_val is None:
            bounds = self.ff.mesh.bounds
            z_val = 0.5 * (bounds[4] + bounds[5])
        return self.ff.mesh.slice(normal='z', origin=(0, 0, z_val))

    def slice_y(self, y_val=0.0):
        """y = const plane (wall-parallel). y_val=0 → wall surface."""
        return self.ff.mesh.slice(normal='y', origin=(0, y_val, 0))

    def slice_x(self, x_val):
        """x = const plane (cross-section at x/D)."""
        return self.ff.mesh.slice(normal='x', origin=(x_val, 0, 0))


class SnapshotSequence:
    """여러 snapshot을 시간순으로 로드하여 분석.

    Parameters
    ----------
    vtu_pattern : str
        glob pattern, e.g. "results/instantaneous/fc-*.vtu"
    flow_params : dict
    """

    def __init__(self, vtu_pattern, flow_params):
        self.flow_params = flow_params
        self._files = sorted(glob.glob(vtu_pattern))
        if not self._files:
            raise FileNotFoundError(f"No VTU files match pattern: {vtu_pattern}")

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        """idx번째 snapshot → SnapshotAnalyzer"""
        from .core import FlowField
        ff = FlowField(self._files[idx])
        return SnapshotAnalyzer(ff, self.flow_params)

    def times(self):
        """시간 배열 (파일명에서 추출).

        Expects filenames like: case-0.00500-p4.vtu → t=0.005
        """
        times = []
        for f in self._files:
            # Try to extract float from filename
            match = re.search(r'(\d+\.\d+)', f)
            if match:
                times.append(float(match.group(1)))
            else:
                times.append(np.nan)
        return np.array(times)

    def extract_probe_timeseries(self, points):
        """여러 snapshot에서 특정 좌표의 시계열 추출.

        Parameters
        ----------
        points : array-like, shape (N, 3)
            Probe 좌표

        Returns
        -------
        dict: {'t': array(M), 'rho': array(M,N), 'u': array(M,N), ...}
        """
        from .core import FlowField
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        t_arr = self.times()
        n_snaps = len(self._files)
        n_pts = len(points)

        # Determine available fields from first snapshot
        ff0 = FlowField(self._files[0])
        field_names = [n for n in ff0.mesh.point_data
                       if ff0.mesh.point_data[n].ndim == 1]

        result = {'t': t_arr}
        for name in field_names:
            result[name] = np.zeros((n_snaps, n_pts))

        for i, fpath in enumerate(self._files):
            ff = FlowField(fpath)
            probed = ff.mesh.probe(points)
            for name in field_names:
                if name in probed.point_data:
                    result[name][i] = probed.point_data[name]

        return result

    def extract_plane_timeseries(self, normal='z', origin=None):
        """여러 snapshot에서 2D plane 시계열 추출 (SPOD 입력용).

        Returns
        -------
        dict: {'t': array(M), 'points': array(P,3), 'fields': dict of array(M,P)}
        """
        from .core import FlowField

        t_arr = self.times()
        # Extract plane from first snapshot to get point layout
        ff0 = FlowField(self._files[0])
        if origin is None:
            bounds = ff0.mesh.bounds
            idx = {'x': 0, 'y': 2, 'z': 4}[normal]
            origin_val = 0.5 * (bounds[idx] + bounds[idx + 1])
            origin = [0, 0, 0]
            origin[idx // 2] = origin_val

        plane0 = ff0.mesh.slice(normal=normal, origin=origin)
        pts = plane0.points.copy()
        n_pts = len(pts)

        field_names = [n for n in plane0.point_data
                       if plane0.point_data[n].ndim == 1]

        fields = {name: np.zeros((len(self._files), n_pts)) for name in field_names}

        for i, fpath in enumerate(self._files):
            ff = FlowField(fpath)
            plane = ff.mesh.slice(normal=normal, origin=origin)
            for name in field_names:
                if name in plane.point_data:
                    fields[name][i] = plane.point_data[name]

        return {'t': t_arr, 'points': pts, 'fields': fields}

    def phase_average(self, frequency, n_phases=8):
        """Phase-averaging at given frequency.

        예: jet flapping frequency → phase-resolved flow field
        """
        raise NotImplementedError("Phase averaging requires sufficient snapshot data")
