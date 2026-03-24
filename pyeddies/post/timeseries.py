"""Time series analysis — PSD, cross-correlation, pre-processing for SPOD/DMD."""
import numpy as np


class ProbeTimeSeries:
    """단일 probe의 시계열 데이터.

    Parameters
    ----------
    t : array, shape (N,)
        Time stamps
    data : dict
        {'u': array(N), 'v': array(N), 'p': array(N), ...}
    name : str, optional
        Probe identifier
    """

    def __init__(self, t, data, name=None):
        self.t = np.asarray(t, dtype=float)
        self.data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.name = name
        self.dt = np.mean(np.diff(self.t))
        self.fs = 1.0 / self.dt  # sampling frequency

    def psd(self, variable='u', method='welch', nperseg=256):
        """Power Spectral Density.

        Returns
        -------
        f : array — frequencies [Hz]
        Pxx : array — power spectral density
        """
        from scipy.signal import welch
        return welch(self.data[variable], fs=self.fs, nperseg=nperseg)

    def strouhal(self, variable='u', D=0.01, U_ref=138.49, **kwargs):
        """PSD in Strouhal number (St = f*D/U).

        Returns
        -------
        St : array
        Pxx : array
        """
        f, Pxx = self.psd(variable, **kwargs)
        St = f * D / U_ref
        return St, Pxx

    def cross_correlation(self, var1, var2, maxlag=None):
        """Cross-correlation R_{12}(τ).

        Parameters
        ----------
        var1, var2 : str
            Variable names in self.data
        maxlag : int, optional
            Maximum lag in samples. Default: N//2.

        Returns
        -------
        lags : array — lag in time units
        R12 : array — normalized cross-correlation
        """
        x = self.data[var1] - np.mean(self.data[var1])
        y = self.data[var2] - np.mean(self.data[var2])
        N = len(x)
        if maxlag is None:
            maxlag = N // 2

        R = np.correlate(x, y, mode='full')
        R = R / (np.std(x) * np.std(y) * N)

        mid = N - 1
        R12 = R[mid - maxlag:mid + maxlag + 1]
        lags = np.arange(-maxlag, maxlag + 1) * self.dt
        return lags, R12

    def autocorrelation(self, variable='u'):
        """Autocorrelation R(τ) → integral time scale.

        Returns
        -------
        lags : array — lag in time units
        R : array — autocorrelation coefficient
        """
        return self.cross_correlation(variable, variable)

    def integral_time_scale(self, variable='u'):
        """T_int = ∫₀^∞ R(τ) dτ  (first zero-crossing).

        Returns
        -------
        T_int : float — integral time scale
        """
        lags, R = self.autocorrelation(variable)
        # Use only positive lags
        mask = lags >= 0
        lags_pos = lags[mask]
        R_pos = R[mask]

        # Find first zero crossing
        zero_idx = np.where(R_pos <= 0)[0]
        if len(zero_idx) > 0:
            idx = zero_idx[0]
        else:
            idx = len(R_pos)

        T_int = np.trapezoid(R_pos[:idx], lags_pos[:idx])
        return T_int


class PlaneTimeSeries:
    """2D plane 시계열 (SPOD/DMD 입력용).

    Parameters
    ----------
    t : array, shape (N_t,)
    points : array, shape (N_pts, 3)
    fields : dict of array(N_t, N_pts)
    """

    def __init__(self, t, points, fields):
        self.t = np.asarray(t, dtype=float)
        self.points = np.asarray(points, dtype=float)
        self.fields = {k: np.asarray(v, dtype=float) for k, v in fields.items()}
        self.dt = np.mean(np.diff(self.t))
        self.n_times = len(self.t)
        self.n_points = self.points.shape[0]

    def snapshot_matrix(self, variables=None):
        """SPOD/DMD 입력 행렬 생성.

        Parameters
        ----------
        variables : list of str
            예: ['u', 'v', 'p'] → 각 변수를 concat

        Returns
        -------
        X : array, shape (N_vars * N_pts, N_t)
            Snapshot matrix (각 열 = 한 시간 스냅샷)
        """
        if variables is None:
            variables = list(self.fields.keys())
        rows = [self.fields[v].T for v in variables]  # each (N_pts, N_t)
        return np.vstack(rows)

    def subtract_mean(self):
        """시간 평균 제거 (fluctuation only) — SPOD/DMD 전처리."""
        for key in self.fields:
            mean = self.fields[key].mean(axis=0)
            self.fields[key] = self.fields[key] - mean

    def to_spod_input(self, variables=None):
        """SPOD 입력 형태로 변환.

        Returns
        -------
        dict: {'X': snapshot_matrix, 'dt': dt, 'points': points}
        """
        X = self.snapshot_matrix(variables)
        return {'X': X, 'dt': self.dt, 'points': self.points}
