"""Film Cooling specific post-processing."""
import numpy as np


class FilmCoolingSweep:
    """StreamwiseSweep 확장 — FC 전용 메트릭 추가.

    Parameters
    ----------
    ff : FlowField
        tavg VTU 로드된 FlowField
    x_stations : array-like
        x/D stations for profile extraction
    flow_params : dict
        Must include: Taw, Tc, R_gas, D, BR, rhoe, ue
    """

    def __init__(self, ff, x_stations, flow_params):
        from .sweep import StreamwiseSweep
        self._sweep = StreamwiseSweep(ff, x_stations, flow_params)
        self.ff = ff
        self.x_stations = np.asarray(x_stations)
        self.flow_params = flow_params
        self.D = flow_params['D']
        self.Taw = flow_params['Taw']
        self.Tc = flow_params['Tc']

    def compute_eta(self, z_mid=None):
        """Spanwise-averaged adiabatic effectiveness η(x/D).

        η = (Taw - T_wall) / (Taw - Tc)

        T_wall from tavg: avg-T at y=0 (wall-adjacent).
        """
        raise NotImplementedError("FC data required — implement after FC run completes")

    def compute_laterally_averaged_eta(self):
        """η_lat(x/D) — spanwise (z) averaged at each x station."""
        raise NotImplementedError

    def compute_centerline_eta(self, z_mid=None):
        """η_cl(x/D) — centerline (z=z_mid) at each x station."""
        raise NotImplementedError

    def compute_br_from_tavg(self, hole_exit_y=0.0):
        """Compute effective BR from time-averaged velocity at hole exit.

        BR = <rho*v>_hole / (rhoe * ue)
        """
        raise NotImplementedError

    def jet_trajectory(self, z_mid=None, threshold=0.5):
        """Extract jet centerline trajectory.

        Parameters
        ----------
        threshold : float
            η threshold for jet boundary (default 0.5)
        """
        raise NotImplementedError

    def tier2_fc_summary(self):
        """FC-specific Tier 2 summary.

        Prints:
        - η_lat at x/D = 1, 3, 5, 10, 15
        - Effective BR
        - Jet penetration height at x/D = 3, 5
        - Comparison with Gritsch data (if available)
        """
        raise NotImplementedError
