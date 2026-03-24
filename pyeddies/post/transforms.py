"""transforms.py — Compressibility transformations.

Van Driest velocity transformation and Crocco-Busemann temperature relation.
Essential for comparing compressible CTBL results with incompressible DNS.
"""
import numpy as np


def van_driest_transform(y_plus, u_plus, rho, rho_w):
    """Van Driest velocity transformation.

    Transforms compressible u+ into an equivalent incompressible u_VD+
    that can be directly compared with Schlatter & Orlu (2010) DNS.

    u_VD+[i] = sum_{j=1}^{i} sqrt(rho[j]/rho_w) * (u+[j] - u+[j-1])

    Reference: Van Driest (1951), Wenzel et al. (2018) JFM

    Parameters
    ----------
    y_plus : np.ndarray
        Wall-normal coordinate in wall units.
    u_plus : np.ndarray
        Velocity in wall units.
    rho : np.ndarray
        Density profile (same grid as y_plus).
    rho_w : float
        Wall density.

    Returns
    -------
    u_vd_plus : np.ndarray
        Van Driest transformed velocity.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    rho = np.asarray(rho, dtype=float)

    u_vd = np.zeros_like(u_plus)
    for i in range(1, len(u_plus)):
        rho_ratio = np.sqrt(rho[i] / rho_w)
        du = u_plus[i] - u_plus[i - 1]
        u_vd[i] = u_vd[i - 1] + rho_ratio * du

    return u_vd


def crocco_busemann(u, u_e, T_e, T_w=None, gamma=1.4, Me=None,
                    r=None, Pr=0.71):
    """Crocco-Busemann temperature-velocity relation (analytical).

    T/T_e = T_w/T_e + (1 - T_w/T_e)(u/u_e) + r*(gamma-1)/2 * Me^2 * (u/u_e)(1 - u/u_e)

    For adiabatic wall:
        T_w = T_aw = T_e * (1 + r*(gamma-1)/2 * Me^2)

    Parameters
    ----------
    u : np.ndarray
        Velocity profile [m/s].
    u_e : float
        Edge velocity [m/s].
    T_e : float
        Edge temperature [K].
    T_w : float or None
        Wall temperature [K]. If None, uses adiabatic wall.
    gamma : float
        Ratio of specific heats.
    Me : float or None
        Edge Mach number. If None, computed from u_e and gamma/T_e.
    r : float or None
        Recovery factor. If None, r = Pr^(1/3).
    Pr : float
        Prandtl number.

    Returns
    -------
    T_cb : np.ndarray
        Crocco-Busemann predicted temperature profile [K].
    """
    u = np.asarray(u, dtype=float)

    if r is None:
        r = Pr ** (1.0 / 3.0)

    if Me is None:
        # a_e = sqrt(gamma * R * T_e), but we don't have R here
        # Use Me = u_e / a_e, but need a_e. Better to require Me.
        raise ValueError("Me must be provided for Crocco-Busemann relation.")

    if T_w is None:
        T_w = T_e * (1.0 + r * (gamma - 1.0) / 2.0 * Me**2)

    u_ratio = u / u_e

    T_cb = T_e * (
        T_w / T_e
        + (1.0 - T_w / T_e) * u_ratio
        + r * (gamma - 1.0) / 2.0 * Me**2 * u_ratio * (1.0 - u_ratio)
    )

    return T_cb


def semi_local_scaling(y, u_tau, rho, mu, rho_w, mu_w):
    """Semi-local wall units (Trettel & Larsson 2016).

    y* = y * u_tau_star / nu_star
    where u_tau_star = sqrt(tau_w / rho), nu_star = mu / rho

    Parameters
    ----------
    y : np.ndarray
        Wall-normal coordinate.
    u_tau : float
        Friction velocity.
    rho, mu : np.ndarray
        Local density and viscosity profiles.
    rho_w, mu_w : float
        Wall values.

    Returns
    -------
    y_star : np.ndarray
        Semi-local wall-normal coordinate.
    """
    y = np.asarray(y, dtype=float)
    rho = np.asarray(rho, dtype=float)
    mu = np.asarray(mu, dtype=float)

    tau_w = rho_w * u_tau**2
    u_tau_star = np.sqrt(tau_w / rho)
    nu_star = mu / rho
    y_star = y * u_tau_star / nu_star

    return y_star
