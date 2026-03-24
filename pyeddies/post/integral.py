"""integral.py — Boundary-layer integral quantities.

delta_99, momentum thickness, displacement thickness, shape factor, Re_theta, Re_tau.
delta_99 logic preserved from original SliceStats.delta_99().
"""
import numpy as np


def delta_99(y, u, u_e=None, frac=0.99, n_edge=10):
    """Boundary-layer thickness (99% of edge velocity).

    Preserved from original SliceStats.delta_99().

    Parameters
    ----------
    y : np.ndarray
        Wall-normal coordinate (sorted ascending).
    u : np.ndarray
        Streamwise velocity.
    u_e : float or None
        Edge velocity. If None, estimated from last n_edge points.
    frac : float
        Fraction of u_e (default 0.99).
    n_edge : int
        Number of points at the edge for u_e estimation.

    Returns
    -------
    float : delta_99 [m]
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if u_e is None:
        u_e = np.mean(u[-n_edge:])

    target = frac * u_e
    idx = np.where(u >= target)[0]

    if len(idx) == 0:
        return y[-1]

    i = idx[0]
    if i == 0:
        return y[0]

    y1, y2 = y[i - 1], y[i]
    U1, U2 = u[i - 1], u[i]

    dU = U2 - U1
    if dU == 0:
        return y2
    alpha = (target - U1) / dU
    return y1 + alpha * (y2 - y1)


def momentum_thickness(y, u, u_e, rho=None, rho_e=None):
    """Momentum thickness theta.

    Incompressible: theta = integral_0^d99 (u/u_e)(1 - u/u_e) dy
    Compressible:   theta = integral_0^d99 (rho*u)/(rho_e*u_e)(1 - u/u_e) dy

    If rho and rho_e given, uses compressible form.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    u_ratio = u / u_e

    if rho is not None and rho_e is not None:
        rho = np.asarray(rho, dtype=float)
        integrand = (rho * u) / (rho_e * u_e) * (1.0 - u_ratio)
    else:
        integrand = u_ratio * (1.0 - u_ratio)

    return float(np.trapezoid(integrand, y))


def displacement_thickness(y, u, u_e, rho=None, rho_e=None):
    """Displacement thickness delta*.

    Incompressible: delta* = integral_0^d99 (1 - u/u_e) dy
    Compressible:   delta* = integral_0^d99 (1 - rho*u/(rho_e*u_e)) dy
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if rho is not None and rho_e is not None:
        rho = np.asarray(rho, dtype=float)
        integrand = 1.0 - (rho * u) / (rho_e * u_e)
    else:
        integrand = 1.0 - u / u_e

    return float(np.trapezoid(integrand, y))


def shape_factor(delta_star, theta):
    """Shape factor H_12 = delta*/theta."""
    if theta <= 0:
        return float('nan')
    return delta_star / theta


def re_theta(theta, u_e, nu_e):
    """Re_theta = u_e * theta / nu_e."""
    return u_e * theta / nu_e


def re_tau(u_tau, delta, nu_w):
    """Re_tau = u_tau * delta / nu_w."""
    return u_tau * delta / nu_w


def compute_bl_integrals(y, u, rho=None, u_e=None, rho_e=None,
                         nu_e=None, nu_w=None, u_tau=None, n_edge=10):
    """Compute all boundary-layer integral quantities at once.

    Parameters
    ----------
    y, u : arrays
        Wall-normal profile.
    rho : array or None
        Density profile (compressible form if given).
    u_e, rho_e : float or None
        Edge values. Auto-estimated if None.
    nu_e, nu_w : float or None
        Kinematic viscosities (for Re_theta, Re_tau).
    u_tau : float or None
        Friction velocity (for Re_tau).
    n_edge : int
        Points to average for edge estimation.

    Returns
    -------
    dict with: delta_99, theta, delta_star, H12, Re_theta, Re_tau
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if u_e is None:
        u_e = np.mean(u[-n_edge:])
    if rho is not None and rho_e is None:
        rho = np.asarray(rho, dtype=float)
        rho_e = np.mean(rho[-n_edge:])

    d99 = delta_99(y, u, u_e=u_e, n_edge=n_edge)

    # Truncate to delta_99 for integration
    mask = y <= d99 * 1.05  # slight margin
    y_bl = y[mask]
    u_bl = u[mask]
    rho_bl = rho[mask] if rho is not None else None

    theta_val = momentum_thickness(y_bl, u_bl, u_e, rho_bl, rho_e)
    dstar_val = displacement_thickness(y_bl, u_bl, u_e, rho_bl, rho_e)
    H12 = shape_factor(dstar_val, theta_val)

    result = {
        'delta_99': d99,
        'theta': theta_val,
        'delta_star': dstar_val,
        'H12': H12,
    }

    if nu_e is not None:
        result['Re_theta'] = re_theta(theta_val, u_e, nu_e)
    if u_tau is not None and nu_w is not None:
        result['Re_tau'] = re_tau(u_tau, d99, nu_w)

    return result
