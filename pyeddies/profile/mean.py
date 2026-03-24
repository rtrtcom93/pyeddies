import numpy as np
from math import log, exp, tanh, sin, sqrt

# ----------------------------
# Reichardt inner law (valid wall → log layer)
# ----------------------------
def reichardt_uplus(yplus, kappa=0.41, B=5.2):
    """Reichardt's formula: u⁺(y⁺), valid from wall to log layer.

    u⁺ = (1/κ)·ln(1+κ·y⁺) + (B - (1/κ)·ln(κ))·(1 - exp(-y⁺/11.6) - (y⁺/8.2)·exp(-y⁺/2.0))

    Reduces to u⁺≈y⁺ as y⁺→0 and to log-law as y⁺→∞.
    """
    yp = np.asarray(yplus, dtype=float)
    return ((1.0/kappa) * np.log(1.0 + kappa * yp)
            + (B - (1.0/kappa) * np.log(kappa))
            * (1.0 - np.exp(-yp/11.6) - (yp/8.2) * np.exp(-yp/2.0)))


# ----------------------------
# Wake function (Coles, 1956)
# ----------------------------
def coles_wake(eta):
    # eta in [0, 1]
    eta = np.clip(eta, 0.0, 1.0)
    return 2.0 * np.sin(0.5 * np.pi * eta)**2


# -----------------------------------------
# Smooth clipping to avoid a "kink" at ue
# -----------------------------------------
def soft_min(a, b, eps=1e-3):
    # smooth approximation of min(a, b)
    # eps sets transition sharpness in velocity units
    return 0.5*(a + b - np.sqrt((a - b)**2 + eps**2))


# ---------------------------------------------------------
# Build mean profile u(y) using wall-wake log + wake term
# Optionally build T(y), rho(y) using Walz relation
# ---------------------------------------------------------
def u_prof_pyfr(
    yval, 
    ue, 
    d99 = None, 
    utau= None, 
    dnu = None,
    k  = 0.41, C = 5.2,
    epscap  = 1.00000e-03,
    Piw     = 4.50000e-01,
    pi      = 3.141592653589793
):

    utau_0 = utau
    dnu_0  = dnu
    d99_0  = d99
    yval = np.asarray(yval)                   
    u = np.zeros_like(yval)
    
    for i in range(len(yval)):
        y = yval[i]
        u[i] = 0.5*(utau_0*((1.0/k)*log(1.0+k*(y/dnu_0))+(C-(1.0/k)*log(k))*(1.0-exp(-(y/dnu_0)/11.6)-((y/dnu_0)/8.2)*exp(-(y/dnu_0)/2.0))+Piw*2.0*pow(sin(0.5*pi*min(max(y/d99_0,0.0),1.0)),2.0))+ue-sqrt((utau_0*((1.0/k)*log(1.0+k*(y/dnu_0))+(C-(1.0/k)*log(k))*(1.0-exp(-(y/dnu_0)/11.6)-((y/dnu_0)/8.2)*exp(-(y/dnu_0)/2.0))+Piw*2.0*pow(sin(0.5*pi*min(max(y/d99_0,0.0),1.0)),2.0))-ue)*(utau_0*((1.0/k)*log(1.0+k*(y/dnu_0))+(C-(1.0/k)*log(k))*(1.0-exp(-(y/dnu_0)/11.6)-((y/dnu_0)/8.2)*exp(-(y/dnu_0)/2.0))+Piw*2.0*pow(sin(0.5*pi*min(max(y/d99_0,0.0),1.0)),2.0))-ue)+epscap*epscap))
    
    return u

def mean_profile_wall_wake(
    y,
    ue,
    utau,
    nuw,
    kappa=0.41,
    B=5.2,
    Pi=0.45,
    delta=5.0e-3,              # outer scale for wake term; often set ~delta99 target
    clip_to_ue=True,
    smooth_clip=True,
    clip_eps=1e-3,
    inner_law="reichardt",     # "reichardt" | "loglaw"
    # compressible options
    compressible=False,
    Te=None,
    Taw=None,
    Me=None,
    gamma=None,
    R=None,
    pe=None,
    Pr=0.71,
):
    y = np.asarray(y)
    yplus = y * utau / nuw
    # avoid log(0)
    yplus_safe = np.maximum(yplus, 1e-12)

    eta = y / delta
    if inner_law == "reichardt":
        Uplus = reichardt_uplus(yplus_safe, kappa, B) + Pi * coles_wake(eta)
    else:
        Uplus = (1.0/kappa) * np.log(yplus_safe) + B + Pi * coles_wake(eta)

    u = utau * Uplus

    if clip_to_ue:
        if smooth_clip:
            u = soft_min(u, ue, eps=clip_eps)
        else:
            u = np.minimum(u, ue)

    if not compressible:
        return u, None, None

    # --- Walz relation for adiabatic wall temperature profile ---
    # recovery factor r ~ Pr^(1/3)
    if any(v is None for v in [Te, Taw, Me, gamma, R, pe]):
        raise ValueError("compressible=True requires Te, Taw, Me, gamma, R, pe")

    r = Pr**(1.0/3.0)

    # Walz: T/Te = 1 + r*(gamma-1)/2 * Me^2 * (1 - (u/ue)^2)
    T = Te * (1.0 + r * (gamma - 1.0) * 0.5 * Me**2 * (1.0 - (u/ue)**2))

    # enforce wall limit to Taw at u~0 (numerical consistency)
    T = np.maximum(T, 1e-6)
    # If you'd like strict match at wall: you can blend near wall or just set T[0]=Taw when y[0]=0

    rho = pe / (R * T)
    return u, T, rho


# ----------------------------
# delta99 finder from u(y)
# ----------------------------
def delta99_from_profile(y, u, ue, enforce_monotone=True):
    target = 0.99 * ue
    y = np.asarray(y, float)
    u = np.asarray(u, float)

    # sort by y and sort u consistently
    idx = np.argsort(y)
    y = y[idx]
    u = u[idx]

    # if u is not monotone, enforce monotone envelope (optional but robust)
    if enforce_monotone:
        u = np.maximum.accumulate(u)

    mask = u >= target
    if not np.any(mask):
        return np.nan

    j = np.argmax(mask)
    if j == 0:
        return y[0]

    y0, y1 = y[j-1], y[j]
    u0, u1 = u[j-1], u[j]
    if abs(u1 - u0) < 1e-14:
        return y1

    return y0 + (target - u0) * (y1 - y0) / (u1 - u0)


# ------------------------------------------
# momentum thickness theta (compressible form)
# ------------------------------------------
def theta_momentum(y, u, ue, rho=None, rhoe=None):
    y = np.asarray(y)
    u = np.asarray(u)
    if rho is None:
        # incompressible
        integrand = (u/ue) * (1.0 - u/ue)
        return np.trapezoid(integrand, y)

    if rhoe is None:
        raise ValueError("compressible theta requires rhoe")

    integrand = (rho * u) / (rhoe * ue) * (1.0 - u/ue)
    return np.trapezoid(integrand, y)


# ---------------------------------------------------
# Solve utau so that delta99 matches a target value
# (keeping ue, nuw, Pi, delta, etc. fixed)
# ---------------------------------------------------
def tune_utau_for_delta99(
    target_delta99,
    ue,
    nuw,
    kappa=0.41,
    B=5.2,
    Pi=0.45,
    delta_for_wake=None,
    utau_lo=1.0,
    utau_hi=30.0,
    ny=12000,
    y_max_factor=150.0,
    clip_to_ue=True,
    smooth_clip=True,
    clip_eps=1e-3,
    inner_law="reichardt",
    compressible=False,
    **comp_kwargs
):
    if delta_for_wake is None:
        delta_for_wake = target_delta99

    y_max = y_max_factor * target_delta99
    y = np.linspace(0.0, y_max, ny)

    def d99_at(utau):
        u, _, _ = mean_profile_wall_wake(
            y=y, ue=ue, utau=utau, nuw=nuw,
            kappa=kappa, B=B, Pi=Pi, delta=delta_for_wake,
            clip_to_ue=clip_to_ue, smooth_clip=smooth_clip, clip_eps=clip_eps,
            inner_law=inner_law,
            compressible=compressible, **comp_kwargs
        )
        return delta99_from_profile(y, u, ue)

    def f(utau):
        d99 = d99_at(utau)
        if np.isnan(d99):
            return np.nan
        return d99 - target_delta99

    # --- ensure utau_lo has a valid crossing ---
    for _ in range(30):
        flo = f(utau_lo)
        if not np.isnan(flo):
            break
        utau_lo *= 1.5
    else:
        raise RuntimeError("Could not find utau_lo that reaches 0.99*ue; check inputs.")

    # --- ensure utau_hi has a valid crossing ---
    for _ in range(30):
        fhi = f(utau_hi)
        if not np.isnan(fhi):
            break
        utau_hi *= 2.0
    else:
        raise RuntimeError("Could not find utau_hi that reaches 0.99*ue; check inputs.")

    # --- now bracket the root (need opposite signs) ---
    if flo * fhi > 0:
        raise RuntimeError(
            f"Root not bracketed: f(utau_lo)={flo:.3e}, f(utau_hi)={fhi:.3e}. "
            "Try widening utau bounds or change Pi/delta_for_wake."
        )

    # bisection
    for _ in range(80):
        um = 0.5*(utau_lo + utau_hi)
        fm = f(um)
        if np.isnan(fm):
            # if mid became invalid (should be rare), push lo up a bit
            utau_lo = um
            continue
        if abs(fm) < 1e-8:
            return um
        if flo * fm <= 0:
            utau_hi = um
            fhi = fm
        else:
            utau_lo = um
            flo = fm

    return 0.5*(utau_lo + utau_hi)