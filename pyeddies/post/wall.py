"""wall.py — Wall shear stress, friction velocity, skin friction.

Quadratic fit logic preserved from original SliceStats.uplus().
"""
import numpy as np


def mu_sutherland(T, mu0=1.716e-5, T0=273.15, S=110.4):
    """Sutherland viscosity model."""
    T = np.asarray(T, float)
    return mu0 * (T / T0) ** 1.5 * ((T0 + S) / (T + S))


def compute_tau_w(y, U, mu_w, method='quadratic_fit'):
    """Compute wall shear stress from near-wall profile.

    Parameters
    ----------
    y : np.ndarray
        Wall-normal coordinates (sorted, ascending from wall).
    U : np.ndarray
        Streamwise velocity profile.
    mu_w : float
        Wall dynamic viscosity [Pa·s].
    method : str
        'quadratic_fit' — U = a*y + b*y² (no-slip), preserved from original.
        'gradient' — simple dU/dy at first point.

    Returns
    -------
    tau_w : float
        Wall shear stress [Pa].
    dUdy_w : float
        Velocity gradient at wall [1/s].
    """
    if len(y) < 2:
        raise RuntimeError("Need at least 2 points for tau_w.")

    if method == 'quadratic_fit':
        dUdy_w = _quadratic_fit_dudy(y, U)
    elif method == 'gradient':
        # Simple first-point gradient
        if y[0] < 1e-12:
            dUdy_w = U[1] / y[1] if y[1] > 0 else 0.0
        else:
            dUdy_w = U[0] / y[0]
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'quadratic_fit' or 'gradient'.")

    tau_w = mu_w * dUdy_w
    return tau_w, dUdy_w


def _quadratic_fit_dudy(y, U):
    """Quadratic fit dU/dy|_w, preserved from original SliceStats.uplus().

    U = a*y + b*y² (no-slip assumption: U(0)=0)
    a = dU/dy|_w
    """
    if len(y) < 3 and y[0] < 1e-12:
        raise RuntimeError("Need at least 3 points with y[0]~0 for quadratic fit.")

    # Case A: y[0] ~ 0 (wall node present) → use indices 1, 2
    if y[0] < 1e-12:
        pts_y = [y[1], y[2]]
        pts_U = [U[1], U[2]]
    # Case B: y[0] is interior point (most PyFR slices) → use indices 0, 1
    else:
        pts_y = [y[0], y[1]]
        pts_U = [U[0], U[1]]

    A = np.array([[pts_y[0], pts_y[0]**2],
                  [pts_y[1], pts_y[1]**2]])
    b_vec = np.array([pts_U[0], pts_U[1]])

    try:
        coeffs = np.linalg.solve(A, b_vec)
        dUdy_w = coeffs[0]
    except np.linalg.LinAlgError:
        dUdy_w = pts_U[0] / pts_y[0]

    return dUdy_w


def compute_u_tau(tau_w, rho_w):
    """Friction velocity: u_tau = sqrt(tau_w / rho_w)."""
    return np.sqrt(abs(tau_w) / rho_w)


def compute_cf(tau_w, rho_e, u_e):
    """Skin friction coefficient: Cf = 2*tau_w / (rho_e * u_e^2)."""
    return 2.0 * tau_w / (rho_e * u_e**2)


def wall_units(y, U, u_tau, nu_w):
    """Convert to wall units.

    Returns
    -------
    dict with 'y_plus', 'u_plus'.
    """
    y_plus = np.asarray(y) * u_tau / nu_w
    u_plus = np.asarray(U) / u_tau
    return {'y_plus': y_plus, 'u_plus': u_plus}


def compute_wall_properties_lagrange(profile, elements, prof_nodes,
                                     R_gas=287.0,
                                     mu0=1.716e-5, T0=273.15, S=110.4,
                                     T_w_override=None):
    """Compute wall properties using Lagrange analytical wall gradient.

    Same output as compute_wall_properties, but uses 5-point 4th-order
    analytical derivative at the wall instead of 2-point quadratic fit.

    Parameters
    ----------
    profile : WallNormalProfile
        Lagrange-interpolated profile (from extract_profile_lagrange).
    elements : list[dict]
        Element map (from extract_profile_lagrange).
    prof_nodes : WallNormalProfile
        Original node-level profile (from extract_profile_lagrange).
        Used for wall gradient computation with original VTU node values.
    R_gas, mu0, T0, S : float
        Thermodynamic parameters.
    T_w_override : float or None
        Override wall temperature.

    Returns
    -------
    dict — same keys as compute_wall_properties.
    """
    from .interp import wall_gradient as _wall_gradient

    y = profile.y
    U_mean = profile.u.copy()
    rho_mean = profile.rho
    p_mean = profile.p

    T_mean = p_mean / (rho_mean * R_gas)
    mu_mean = mu_sutherland(T_mean, mu0, T0, S)

    y_w = y[0]
    rho_w = rho_mean[0]

    if T_w_override is not None:
        T_w = float(T_w_override)
    else:
        T_w = T_mean[0]

    mu_w = mu_sutherland(T_w, mu0, T0, S)

    # Wall gradient from original VTU node values (5-point 4th-order)
    dUdy_w = _wall_gradient(elements, prof_nodes.u)
    tau_w = mu_w * dUdy_w
    u_tau = compute_u_tau(tau_w, rho_w)
    nu_w = mu_w / rho_w

    U_mean_out = U_mean.copy()
    if y_w < 1e-9:
        U_mean_out[0] = 0.0

    y_plus = y * u_tau / nu_w
    u_plus = U_mean_out / u_tau

    return {
        'y': y, 'U_mean': U_mean_out, 'rho_mean': rho_mean,
        'T_mean': T_mean, 'mu_mean': mu_mean,
        'y_plus': y_plus, 'u_plus': u_plus,
        'y_w': y_w, 'T_w': T_w, 'mu_w': mu_w, 'rho_w': rho_w,
        'dUdy_w': dUdy_w, 'tau_w': tau_w, 'u_tau': u_tau, 'nu_w': nu_w,
    }


def compute_wall_properties(profile, R_gas=287.0,
                            mu0=1.716e-5, T0=273.15, S=110.4,
                            T_w_override=None, method='quadratic_fit'):
    """Compute all wall properties from a WallNormalProfile.

    Combines the full logic of the original SliceStats.uplus().

    Parameters
    ----------
    profile : WallNormalProfile
        Must have fields: u, rho, p (and optionally T).
    R_gas, mu0, T0, S : float
        Thermodynamic parameters.
    T_w_override : float or None
        Override wall temperature.
    method : str
        'quadratic_fit' or 'gradient'.

    Returns
    -------
    dict with: y, U_mean, rho_mean, T_mean, mu_mean,
               y_plus, u_plus, y_w, T_w, mu_w, rho_w,
               dUdy_w, tau_w, u_tau, nu_w
    """
    y = profile.y
    U_mean = profile.u.copy()
    rho_mean = profile.rho
    p_mean = profile.p

    T_mean = p_mean / (rho_mean * R_gas)
    mu_mean = mu_sutherland(T_mean, mu0, T0, S)

    if len(y) < 3:
        raise RuntimeError("Need at least 3 points in wall-normal direction.")

    y_w = y[0]
    rho_w = rho_mean[0]

    if T_w_override is not None:
        T_w = float(T_w_override)
    else:
        T_w = T_mean[0]

    mu_w = mu_sutherland(T_w, mu0, T0, S)
    tau_w, dUdy_w = compute_tau_w(y, U_mean, mu_w, method=method)
    u_tau = compute_u_tau(tau_w, rho_w)
    nu_w = mu_w / rho_w

    # Visualization: set wall point to zero if close to wall
    U_mean_out = U_mean.copy()
    if y_w < 1e-9:
        U_mean_out[0] = 0.0

    y_plus = y * u_tau / nu_w
    u_plus = U_mean_out / u_tau

    return {
        'y': y, 'U_mean': U_mean_out, 'rho_mean': rho_mean,
        'T_mean': T_mean, 'mu_mean': mu_mean,
        'y_plus': y_plus, 'u_plus': u_plus,
        'y_w': y_w, 'T_w': T_w, 'mu_w': mu_w, 'rho_w': rho_w,
        'dUdy_w': dUdy_w, 'tau_w': tau_w, 'u_tau': u_tau, 'nu_w': nu_w,
    }
