"""turbstats.py — Turbulence statistics: Reynolds/Favre decomposition, TKE.

Extracts fluctuation components from time-averaged fields.
"""
import numpy as np


def reynolds_stress(profile):
    """Reynolds decomposition from tavg fields.

    R_ij = <u_i u_j> - <u_i><u_j>

    Required profile fields: u, v, w, uu, vv, ww, uv
    Optional: uw, vw

    Parameters
    ----------
    profile : WallNormalProfile
        Must have tavg fields.

    Returns
    -------
    dict: 'uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'tke'
        Reynolds stress components <u_i'u_j'>.
    """
    u = profile.u
    v = profile.v
    w = profile.w

    result = {}

    if profile.has_field('uu'):
        result['uu'] = profile.uu - u * u
    if profile.has_field('vv'):
        result['vv'] = profile.vv - v * v
    if profile.has_field('ww'):
        result['ww'] = profile.ww - w * w
    if profile.has_field('uv'):
        result['uv'] = profile.uv - u * v
    if profile.has_field('uw'):
        result['uw'] = profile.uw - u * w
    if profile.has_field('vw'):
        result['vw'] = profile.vw - v * w

    # TKE = 0.5 * (R_11 + R_22 + R_33)
    if 'uu' in result and 'vv' in result and 'ww' in result:
        result['tke'] = 0.5 * (result['uu'] + result['vv'] + result['ww'])

    return result


def favre_stress(profile):
    """Favre decomposition from tavg fields (compressible).

    Favre average: u_tilde = <rho*u> / <rho>
    Favre stress:  R_tilde_ij = <rho*u_i*u_j>/<rho> - u_tilde_i * u_tilde_j

    Required profile fields: rho, rhou, rhov, rhow, rhouu, rhovv, rhoww, rhouv
    Optional: rhouw, rhovw

    Parameters
    ----------
    profile : WallNormalProfile

    Returns
    -------
    dict: 'u_favre', 'v_favre', 'w_favre',
          'uu_favre', 'vv_favre', 'ww_favre', 'uv_favre', ...
          'tke_favre'
    """
    rho = profile.rho
    result = {}

    # Favre-averaged velocities
    if profile.has_field('rhou'):
        u_f = profile.rhou / rho
        result['u_favre'] = u_f
    if profile.has_field('rhov'):
        v_f = profile.rhov / rho
        result['v_favre'] = v_f
    if profile.has_field('rhow'):
        w_f = profile.rhow / rho
        result['w_favre'] = w_f

    # Favre stresses
    if profile.has_field('rhouu') and 'u_favre' in result:
        result['uu_favre'] = profile.rhouu / rho - result['u_favre']**2
    if profile.has_field('rhovv') and 'v_favre' in result:
        result['vv_favre'] = profile.rhovv / rho - result['v_favre']**2
    if profile.has_field('rhoww') and 'w_favre' in result:
        result['ww_favre'] = profile.rhoww / rho - result['w_favre']**2
    if profile.has_field('rhouv') and 'u_favre' in result and 'v_favre' in result:
        result['uv_favre'] = profile.rhouv / rho - result['u_favre'] * result['v_favre']
    if profile.has_field('rhouw') and 'u_favre' in result and 'w_favre' in result:
        result['uw_favre'] = profile.rhouw / rho - result['u_favre'] * result['w_favre']
    if profile.has_field('rhovw') and 'v_favre' in result and 'w_favre' in result:
        result['vw_favre'] = profile.rhovw / rho - result['v_favre'] * result['w_favre']

    # Favre TKE
    if all(k in result for k in ('uu_favre', 'vv_favre', 'ww_favre')):
        result['tke_favre'] = 0.5 * (
            result['uu_favre'] + result['vv_favre'] + result['ww_favre']
        )

    return result


def turbulence_intensity(tke, u_e):
    """Turbulence intensity: TI = sqrt(2k/3) / u_e."""
    return np.sqrt(2.0 * np.asarray(tke) / 3.0) / u_e


def normalize_reynolds_stress(R_dict, u_tau):
    """Normalize Reynolds stresses by u_tau^2.

    Parameters
    ----------
    R_dict : dict
        Output from reynolds_stress() or favre_stress().
    u_tau : float
        Friction velocity.

    Returns
    -------
    dict with same keys, values divided by u_tau^2.
    """
    ut2 = u_tau ** 2
    result = {}
    for key, val in R_dict.items():
        if isinstance(val, np.ndarray) and key not in ('u_favre', 'v_favre', 'w_favre'):
            result[key] = val / ut2
        else:
            result[key] = val
    return result
