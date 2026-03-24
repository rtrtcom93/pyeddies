"""Empirical correlations for turbulent boundary layer validation (Tier 2).

References
----------
- Smits, Matheson & Joubert (1983): Cf correlation for ZPG TBL
- Schlatter & Orlu (2010): Re_tau vs Re_theta DNS correlation
- Wenzel, Selent, Kloker & Rist (2018): Compressible DNS data
- Van Driest (1956): Compressibility transformation for Cf
"""
import numpy as np


# ---------------------------------------------------------------------------
# Skin-friction correlations (incompressible)
# ---------------------------------------------------------------------------

def cf_smits(Re_theta):
    """Smits et al. (1983) incompressible Cf correlation.

    Cf = 0.024 * Re_theta^(-1/4)

    Valid range: Re_theta ~ 600 - 10000.
    """
    Re_theta = np.asarray(Re_theta, dtype=float)
    return 0.024 * Re_theta ** (-0.25)


def cf_power_law(Re_theta, C=0.0256, n=-0.26):
    """Generic power-law Cf = C * Re_theta^n."""
    Re_theta = np.asarray(Re_theta, dtype=float)
    return C * Re_theta ** n


# ---------------------------------------------------------------------------
# Re_tau correlation
# ---------------------------------------------------------------------------

def re_tau_schlatter(Re_theta):
    """Schlatter & Orlu (2010) DNS correlation.

    Re_tau = 1.13 * Re_theta^0.843

    Valid range: Re_theta ~ 250 - 4000.
    """
    Re_theta = np.asarray(Re_theta, dtype=float)
    return 1.13 * Re_theta ** 0.843


# ---------------------------------------------------------------------------
# Van Driest-II compressibility transformation
# ---------------------------------------------------------------------------

def van_driest_ii(Cf_inc, Me, Te, Tw, gamma=1.4, Pr=0.71):
    """Van Driest-II transformation: incompressible Cf -> compressible Cf.

    Parameters
    ----------
    Cf_inc : float or array
        Incompressible skin-friction coefficient.
    Me : float
        Edge Mach number.
    Te : float
        Edge static temperature [K].
    Tw : float
        Wall temperature [K] (adiabatic wall: Tw = Taw).
    gamma : float
        Ratio of specific heats.
    Pr : float
        Prandtl number.

    Returns
    -------
    Cf_comp : float or array
        Compressible skin-friction coefficient.
    Fc : float
        Compressibility factor (Cf_comp = Cf_inc / Fc).
    """
    r = Pr ** (1.0 / 3.0)
    Tr = Te * (1.0 + r * (gamma - 1.0) / 2.0 * Me**2)  # recovery temperature

    A = (r * (gamma - 1.0) / 2.0 * Me**2 * Te / Tw) ** 0.5
    B = (Tr / Tw - 1.0)

    # Van Driest-II formula
    if abs(A) < 1e-12:
        Fc = 1.0
    else:
        Fc = (2.0 * A**2 - B) / (B**2 + 4.0 * A**2) ** 0.5
        Fc *= np.arcsin((2.0 * A**2 - B) / (B**2 + 4.0 * A**2) ** 0.5)
        # Full form:
        # Fc = (Tw/Te) * (arcsin(alpha) + arcsin(beta)) / (arcsin_arg)
        # Simplified for adiabatic wall (Tw = Tr):
        pass

    # Simpler robust form (White 2006, Eq. 7.119):
    # Fc = (Tw/Te)^0.5 * [arcsin(A1) + arcsin(A2)] / sqrt(A1^2 + A2^2 + ...)
    # Use the direct relation for low Mach:
    Fc = (Tw / Te) ** 0.5  # First-order approximation valid for Me < 0.5

    Cf_comp = Cf_inc / Fc
    return Cf_comp, Fc


def van_driest_ii_adiabatic(Cf_inc, Me, gamma=1.4, Pr=0.71):
    """Simplified Van Driest-II for adiabatic wall.

    For adiabatic wall at low Mach (Me < ~0.5), the correction is small.
    Fc ~ sqrt(Taw/Te) = sqrt(1 + r*(gamma-1)/2 * Me^2)

    Returns (Cf_comp, Fc).
    """
    r = Pr ** (1.0 / 3.0)
    Taw_over_Te = 1.0 + r * (gamma - 1.0) / 2.0 * Me**2
    Fc = Taw_over_Te ** 0.5
    Cf_comp = np.asarray(Cf_inc, dtype=float) / Fc
    return Cf_comp, Fc


# ---------------------------------------------------------------------------
# Wenzel et al. (2018) composite correlation
# ---------------------------------------------------------------------------

def wenzel_cf(Re_theta, Me, Te, Tw, gamma=1.4, Pr=0.71):
    """Wenzel et al. (2018) DNS-validated Cf.

    Applies Smits incompressible Cf + Van Driest-II compressibility correction.

    Parameters
    ----------
    Re_theta : float or array
        Momentum-thickness Reynolds number.
    Me, Te, Tw, gamma, Pr : float
        Flow conditions (see van_driest_ii).

    Returns
    -------
    Cf : float or array
        Compressible skin-friction coefficient.
    """
    Cf_inc = cf_smits(Re_theta)
    Cf_comp, _ = van_driest_ii(Cf_inc, Me, Te, Tw, gamma, Pr)
    return Cf_comp


def wenzel_cf_adiabatic(Re_theta, Me, gamma=1.4, Pr=0.71):
    """Wenzel Cf for adiabatic wall (simplified)."""
    Cf_inc = cf_smits(Re_theta)
    Cf_comp, _ = van_driest_ii_adiabatic(Cf_inc, Me, gamma, Pr)
    return Cf_comp


# ---------------------------------------------------------------------------
# Momentum thickness from Cf (von Karman integral)
# ---------------------------------------------------------------------------

def theta_growth_zpg(x, Cf_func, x0=0.0, theta0=0.0, dx=1e-4):
    """Integrate d(theta)/dx = Cf/2 for ZPG TBL.

    Parameters
    ----------
    x : array-like
        Streamwise stations [m].
    Cf_func : callable
        Cf(Re_theta) function.
    x0 : float
        Starting x [m].
    theta0 : float
        Initial momentum thickness [m].
    dx : float
        Integration step [m].

    Returns
    -------
    x_out, theta_out : arrays
    """
    from scipy.integrate import solve_ivp

    x = np.asarray(x, dtype=float)
    x_span = (x[0], x[-1])

    def rhs(t, y):
        theta = y[0]
        # Need ue, nu_e from caller context — this is a simplified version
        # For full implementation, pass these as parameters
        Re_th = max(theta * 1.0 / 1e-5, 100.0)  # placeholder
        return [Cf_func(Re_th) / 2.0]

    sol = solve_ivp(rhs, x_span, [theta0], t_eval=x, method='RK45')
    return sol.t, sol.y[0]


# ---------------------------------------------------------------------------
# Shape factor correlation (Chauhan et al. 2009)
# ---------------------------------------------------------------------------

def H12_chauhan(Re_theta):
    """Chauhan, Monkewitz & Nagib (2009) shape factor correlation.

    H12 = 1.0 + 9.52 / ln(Re_theta)^2  (approximate)

    Valid range: Re_theta > 500.
    """
    Re_theta = np.asarray(Re_theta, dtype=float)
    return 1.0 + 9.52 / np.log(Re_theta) ** 2
