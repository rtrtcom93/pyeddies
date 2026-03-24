#!/usr/bin/env python3
"""
Film Cooling CTBL 파라미터 결정 스크립트

Re_D = 13,000 (Gritsch 1/10) 을 anchor로, 모든 파라미터를 자기일관적으로 역산한다.
이 스크립트 하나로 전체 파라미터 결정 과정을 재현할 수 있다.

사용법:
    python parameter_determination.py
    python parameter_determination.py --Re_D 26000    # 다른 Re_D로 시도
    python parameter_determination.py --Te 288        # Wenzel 조건으로 시도
"""

import numpy as np
import argparse
from math import sqrt, log, exp, sin, pi


# ============================================================
# 1. NASA-9 다항식 (공기, mass-basis)
# ============================================================

# 공기 NASA-9 계수 (mass-basis, J/kg/K)
AIR_NASA9_LOW = [2898903, -56496.26, 1437.799, -1.653609, 
                 0.003062254, -2.2791387e-6, 6.272365e-10]
AIR_NASA9_HIGH = [6.932484e7, -361053.2, 1476.665, -0.06138349,
                  2.027963e-5, -3.075525e-9, 1.888054e-13]
AIR_T_MID = 1000.0
AIR_M = 28.97e-3  # kg/mol
R_UNIVERSAL = 8.31446261815324  # J/mol/K


def nasa9_cp_mass(T, low=AIR_NASA9_LOW, high=AIR_NASA9_HIGH, Tmid=AIR_T_MID):
    """NASA-9 다항식으로 Cp [J/kg/K] 계산 (mass-basis 계수)"""
    c = low if T <= Tmid else high
    a0, a1, a2, a3, a4, a5, a6 = c
    return a0*T**-2 + a1*T**-1 + a2 + a3*T + a4*T**2 + a5*T**3 + a6*T**4


def nasa9_gamma(T):
    """NASA-9에서 γ = Cp/Cv 계산"""
    cp_mass = nasa9_cp_mass(T)
    R_specific = R_UNIVERSAL / AIR_M
    cv_mass = cp_mass - R_specific
    return cp_mass / cv_mass


# ============================================================
# 2. Sutherland 점성 법칙
# ============================================================

def mu_sutherland(T, mu0=1.716e-5, T0=273.15, S=110.4):
    """Sutherland 법칙으로 동점성계수 계산 [Pa·s]"""
    return mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)


# ============================================================
# 3. Phase 1: 기본 열역학 (Te만으로 결정)
# ============================================================

def phase1_thermodynamics(Te, Me, Pr=0.71):
    """Te, Me로부터 기본 열역학 변수 계산"""
    gamma = float(nasa9_gamma(Te))
    cp = float(nasa9_cp_mass(Te))
    R = cp * (gamma - 1) / gamma
    a = sqrt(gamma * R * Te)
    ue = Me * a
    mu_e = mu_sutherland(Te)
    
    return {
        'Te': Te, 'Me': Me, 'Pr': Pr,
        'gamma': gamma, 'cp': cp, 'R': R,
        'a': a, 'ue': ue, 'mu_e': mu_e,
    }


# ============================================================
# 4. Phase 2: pe 역산 (Re_D가 anchor)
# ============================================================

def phase2_pressure(thermo, Re_D, D):
    """Re_D로부터 pe 역산"""
    ue = thermo['ue']
    mu_e = thermo['mu_e']
    R = thermo['R']
    Te = thermo['Te']
    
    rho_e = Re_D * mu_e / (ue * D)
    pe = rho_e * R * Te
    nu_e = mu_e / rho_e
    
    # 역검증: Re_D 재계산
    Re_D_check = rho_e * ue * D / mu_e
    
    return {
        'Re_D': Re_D, 'D': D,
        'rho_e': rho_e, 'pe': pe, 'nu_e': nu_e,
        'Re_D_check': Re_D_check,
    }


# ============================================================
# 5. Phase 3: 경계층 파라미터
# ============================================================

def phase3_boundary_layer(thermo, pressure, d99_over_D):
    """δ99/D 비율로부터 경계층 파라미터 계산"""
    D = pressure['D']
    d99 = d99_over_D * D
    Re_d99 = pressure['rho_e'] * thermo['ue'] * d99 / thermo['mu_e']
    
    return {
        'd99': d99, 'd99_over_D': d99_over_D,
        'Re_d99': Re_d99,
    }


# ============================================================
# 6. Phase 4: 벽면 물성
# ============================================================

def phase4_wall_properties(thermo, pressure):
    """등엔트로피 관계 + Walz로 벽면 물성 계산"""
    Te = thermo['Te']
    Me = thermo['Me']
    gamma = thermo['gamma']
    R = thermo['R']
    Pr = thermo['Pr']
    pe = pressure['pe']
    
    Tte = Te * (1 + (gamma - 1) / 2 * Me**2)
    pte = pe * (Tte / Te) ** (gamma / (gamma - 1))
    Taw = Te + Pr**(1/3) * (Tte - Te)
    
    rho_w = pe / (R * Taw)  # pw ≈ pe
    mu_w = mu_sutherland(Taw)
    nu_w = mu_w / rho_w
    
    return {
        'Tte': Tte, 'pte': pte, 'Taw': Taw,
        'rho_w': rho_w, 'mu_w': mu_w, 'nu_w': nu_w,
    }


# ============================================================
# 7. Phase 5: PyFR constants + 마찰 속도 추정
# ============================================================

def phase5_derived(thermo, pressure, wall, bl):
    """cpTref, cpTs, u_τ 추정, Re_τ, Re_θ 추정"""
    gamma = thermo['gamma']
    a = thermo['a']
    ue = thermo['ue']
    d99 = bl['d99']
    nu_w = wall['nu_w']
    rho_w = wall['rho_w']
    rho_e = pressure['rho_e']
    
    # Sutherland 파라미터 (PyFR 형식)
    T0_suth = 273.15
    S_suth = 110.4
    cpTref = a**2 / (gamma - 1)
    cpTs = cpTref * S_suth / T0_suth
    
    # Cf 추정 (비압축성 상관식 + van Driest-II 보정은 간략화)
    # Re_θ를 먼저 추정: Re_d99 ≈ 8 × Re_θ (경험적)
    Re_d99 = bl['Re_d99']
    Re_theta_est = Re_d99 / 8.0
    
    # Cf 비압축성 상관식
    cf_inc = 0.024 * Re_theta_est**(-0.25) if Re_theta_est > 0 else 0.005
    
    # van Driest-II 보정 (압축성)
    Te, Taw = thermo['Te'], wall['Taw']
    Taw_Te = Taw / Te
    if Taw_Te > 1.0:
        A_vd = (Taw_Te - 1) / sqrt(Taw_Te * (Taw_Te - 1))
        Fc = (Taw_Te - 1) / np.arcsin(A_vd) if abs(np.arcsin(A_vd)) > 1e-10 else 1.0
    else:
        Fc = 1.0
    
    mu_e = thermo['mu_e']
    mu_w_val = wall['mu_w']
    cf_comp = (1 / Fc) * cf_inc  # 간략화된 보정
    
    # u_τ 추정
    u_tau_est = ue * sqrt(cf_comp / 2)
    
    # Re_τ, d_ν
    Re_tau_est = rho_w * u_tau_est * d99 / mu_w_val
    d_nu = nu_w / u_tau_est
    
    # Re_θ 상관식에서 역산 (Schlatter & Örlü 2010)
    # Re_τ = 1.13 × Re_θ^0.843  →  Re_θ = (Re_τ / 1.13)^(1/0.843)
    Re_theta_from_tau = (Re_tau_est / 1.13) ** (1 / 0.843) if Re_tau_est > 0 else 0
    
    return {
        'cpTref': cpTref, 'cpTs': cpTs,
        'cf_est': cf_comp, 'u_tau_est': u_tau_est,
        'Re_tau_est': Re_tau_est, 'Re_theta_est': Re_theta_from_tau,
        'd_nu': d_nu,
    }


# ============================================================
# 8. Phase 6: Coolant 물성 (Film Cooling용)
# ============================================================

def phase6_coolant_jet(thermo, pressure, Tc_over_Tm, BR, D,
                       Cd_mode='gritsch', max_iter=50, tol=1e-4):
    """Coolant 물성 + BR 무차원수 + Gritsch Cd + Pt,c 역산.
 
    Parameters
    ----------
    thermo   : dict from phase1_thermodynamics
    pressure : dict from phase2_pressure
    Tc_over_Tm : float, temperature ratio (0.54)
    BR       : float, target blowing ratio
    D        : float, hole diameter [m]
    Cd_mode  : 'gritsch', 'nominal' (Cd=1.0), or float
 
    Returns
    -------
    dict with coolant properties, BR params, and Pt,c estimates.
    """
    Te = thermo['Te']
    gamma_m = thermo['gamma']
    pe = pressure['pe']
    rho_e = pressure['rho_e']
    ue = thermo['ue']
    A_hole = pi / 4 * D**2
 
    # --- Coolant thermodynamics ---
    Tc = Tc_over_Tm * Te
    gamma_c = float(nasa9_gamma(Tc))
    cp_c = float(nasa9_cp_mass(Tc))
    R_c = R_UNIVERSAL / AIR_M           # R is T-independent for air
    rho_c = pe / (R_c * Tc)
    mu_c = mu_sutherland(Tc)
    DR = rho_c / rho_e
 
    delta_gamma = abs(gamma_c - gamma_m) / gamma_m * 100  # Tier 1-F
 
    # --- BR dimensionless numbers ---
    VR = BR / DR
    I = BR * VR                          # = BR² / DR
    Uc = BR * rho_e * ue / rho_c
 
    # --- Target mass flow ---
    mdot_target = BR * rho_e * ue * A_hole
 
    # --- Ttc: plenum is stagnant → Ttc = Tc ---
    Ttc = Tc
 
    # --- I_jet/extCr for Gritsch correlation ---
    I_jet_extCr = rho_c * Uc**2 / (rho_e * ue**2)
 
    # --- Pt,c iteration: find Pt,c such that Cd*mdot_ideal = mdot_target ---
    Ptc = pe + 0.5 * rho_c * Uc**2      # incompressible total-P initial guess
 
    converged = False
    for iteration in range(max_iter):
        PR = Ptc / pe
        mdot_ideal = isentropic_mdot_ideal(Ptc, Ttc, pe, D, gamma_c)
 
        if mdot_ideal <= 0:
            Ptc *= 1.5
            continue
 
        # Cd prediction
        if Cd_mode == 'nominal':
            Cd_pred = 1.0
        elif Cd_mode == 'gritsch':
            Cd_pred = gritsch_cd(PR, I_jet_extCr)
        elif isinstance(Cd_mode, (int, float)):
            Cd_pred = float(Cd_mode)
        else:
            Cd_pred = 0.70
 
        # Actual mdot with this Cd and current Pt,c
        mdot_actual = Cd_pred * mdot_ideal
 
        # Convergence check
        err = abs(mdot_actual - mdot_target) / mdot_target
        if err < tol:
            converged = True
            break
 
        # Update Pt,c: scale (Ptc - pe) proportionally
        Ptc_new = pe + (Ptc - pe) * (mdot_target / mdot_actual)
        if Ptc_new <= pe:
            Ptc_new = pe * 1.001
        Ptc = Ptc_new
 
    # Final values
    PR_final = Ptc / pe
    mdot_ideal_final = isentropic_mdot_ideal(Ptc, Ttc, pe, D, gamma_c)
    Cd_final = mdot_target / mdot_ideal_final if mdot_ideal_final > 0 else 0
    Cd_gritsch_final = gritsch_cd(PR_final, I_jet_extCr)
 
    return {
        # Coolant thermo
        'Tc': Tc, 'Tc_over_Tm': Tc_over_Tm,
        'gamma_c': gamma_c, 'cp_c': cp_c, 'R_c': R_c,
        'rho_c': rho_c, 'mu_c': mu_c, 'DR': DR,
        'delta_gamma_pct': delta_gamma,
        # BR params
        'BR': BR, 'VR': VR, 'I': I, 'Uc': Uc,
        'I_jet_extCr': I_jet_extCr,
        # Mass flow
        'mdot_target': mdot_target,
        'A_hole': A_hole,
        # Ptc results (Gritsch Cd-corrected)
        'Ptc': Ptc, 'PR': PR_final, 'Ttc': Ttc,
        'Cd_gritsch': Cd_gritsch_final,
        'Cd_actual': Cd_final,
        'converged': converged,
        'n_iter': iteration + 1 if converged else max_iter,
    }

def gritsch_cd_nocr(PR):
    """Cd_noCr(PR) — Gritsch (1998 AIAA) Fig.8
    Cylindrical alpha=30°, Mac=0, Mam=0 (plenum-to-plenum baseline).
    Linear interpolation on approximate graph data."""
    PR_data = np.array([1.00, 1.05, 1.10, 1.25, 1.40, 1.60, 2.00, 2.25])
    Cd_data = np.array([0.40, 0.55, 0.60, 0.69, 0.74, 0.78, 0.80, 0.81])
    PR_clip = np.clip(PR, PR_data[0], PR_data[-1])
    return float(np.interp(PR_clip, PR_data, Cd_data))
 
 
def gritsch_f_ext(I_jet_extCr):
    """f_ext = Cd_extCr / Cd_noCr — Gritsch (2001) Fig.7
    Cylindrical alpha=30°. Log-space interpolation."""
    I_data = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    f_data = np.array([0.30, 0.38, 0.45, 0.55, 0.72, 0.80, 0.90, 0.97, 1.00, 1.00, 1.00])
    if I_jet_extCr <= 0:
        return f_data[0]
    logI = np.log10(np.clip(I_jet_extCr, I_data[0], I_data[-1]))
    logI_data = np.log10(I_data)
    return float(np.interp(logI, logI_data, f_data))
 
 
def gritsch_cd(PR, I_jet_extCr):
    """Gritsch (1998 AIAA) Eq.10:
    Cd = Cd_noCr(PR) x f_ext(I_jet/extCr)
    Mac≈0 (plenum feed) → f_int = 1.0"""
    return gritsch_cd_nocr(PR) * gritsch_f_ext(I_jet_extCr)
 
 
def isentropic_mdot_ideal(Ptc, Ttc, pm, D, gamma):
    """1D isentropic ideal mass flow rate [kg/s].
    Gritsch Cd definition denominator."""
    R = R_UNIVERSAL / AIR_M
    A = pi / 4 * D**2
    PR = Ptc / pm
    if PR <= 1.0:
        return 0.0
    term1 = (pm / Ptc) ** ((gamma + 1) / (2 * gamma))
    term2 = sqrt(2 * gamma / ((gamma - 1) * R * Ttc) * (PR**((gamma - 1) / gamma) - 1))
    return Ptc * A * term1 * term2


# ============================================================
# 9. Tier 1 자기일관성 검증
# ============================================================

def tier1_validation(thermo, pressure, bl, wall, derived):
    """Tier 1 검증: 자기일관성 체크"""
    checks = []
    
    # 1-A: 열역학적 일관성
    R_check = R_UNIVERSAL / AIR_M
    R_from_cp = thermo['cp'] * (thermo['gamma'] - 1) / thermo['gamma']
    err_R = abs(R_from_cp - R_check) / R_check * 100
    checks.append(('R consistency', err_R, 0.1, 'PASS' if err_R < 0.1 else 'FAIL'))
    
    rho_check = pressure['pe'] / (thermo['R'] * thermo['Te'])
    err_rho = abs(rho_check - pressure['rho_e']) / pressure['rho_e'] * 100
    checks.append(('rho = pe/(R*Te)', err_rho, 0.1, 'PASS' if err_rho < 0.1 else 'FAIL'))
    
    a_check = sqrt(thermo['gamma'] * thermo['R'] * thermo['Te'])
    err_a = abs(a_check - thermo['a']) / thermo['a'] * 100
    checks.append(('a = sqrt(gRT)', err_a, 0.1, 'PASS' if err_a < 0.1 else 'FAIL'))
    
    Ma_check = thermo['ue'] / thermo['a']
    err_Ma = abs(Ma_check - thermo['Me']) / thermo['Me'] * 100
    checks.append(('Ma = ue/a', err_Ma, 0.1, 'PASS' if err_Ma < 0.1 else 'FAIL'))
    
    Re_D_check = pressure['rho_e'] * thermo['ue'] * pressure['D'] / thermo['mu_e']
    err_Re = abs(Re_D_check - pressure['Re_D']) / pressure['Re_D'] * 100
    checks.append(('Re_D roundtrip', err_Re, 0.1, 'PASS' if err_Re < 0.1 else 'FAIL'))
    
    return checks


# ============================================================
# 10. Wenzel 비교 범위 확인
# ============================================================

def wenzel_range_check(thermo, pressure, wall, d99_list=[3.5e-3, 4.0e-3, 4.5e-3, 5.0e-3]):
    """다양한 d99에서 Re_θ를 추정하고 Wenzel 범위와 비교"""
    results = []
    
    for d99 in d99_list:
        Re_d99 = pressure['rho_e'] * thermo['ue'] * d99 / thermo['mu_e']
        Re_theta_est = Re_d99 / 8.0
        cf_est = 0.024 * Re_theta_est**(-0.25) if Re_theta_est > 0 else 0
        u_tau_est = thermo['ue'] * sqrt(cf_est / 2)
        Re_tau_est = wall['rho_w'] * u_tau_est * d99 / wall['mu_w']
        Re_theta_from_tau = (Re_tau_est / 1.13) ** (1 / 0.843) if Re_tau_est > 0 else 0
        
        d99_D = d99 / pressure['D']
        wenzel_min = 670  # Wenzel Ma=0.3 최저 Re_θ
        in_range = "진입" if Re_theta_from_tau >= wenzel_min else "범위 밖"
        if wenzel_min * 0.95 < Re_theta_from_tau < wenzel_min:
            in_range = "거의 진입"
        
        results.append({
            'd99_mm': d99 * 1000,
            'd99_D': d99_D,
            'u_tau': u_tau_est,
            'Re_tau': Re_tau_est,
            'Re_theta': Re_theta_from_tau,
            'cf': cf_est,
            'wenzel': in_range,
        })
    
    return results


# ============================================================
# 11. 전체 요약 출력
# ============================================================

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_param(name, value, unit="", ref=None, fmt=".5e"):
    line = f"  {name:16s} = {value:{fmt}}  {unit}"
    if ref is not None:
        err = abs(value - ref) / abs(ref) * 100 if ref != 0 else 0
        line += f"  (ref: {ref:{fmt}}, err: {err:.2f}%)"
    print(line)


def main(Te=537.0, Me=0.3, Pr=0.71, Re_D=13000, D=0.010,
         d99_over_D=0.5, Tc_over_Tm=0.54, alpha=30.0, L_over_D=6.0,
         BR=1.0):
    
    print_section("독립 변수 (입력)")
    print(f"  Te            = {Te} K")
    print(f"  Me            = {Me}")
    print(f"  Pr            = {Pr}")
    print(f"  Re_D (anchor) = {Re_D}")
    print(f"  D             = {D*1000} mm")
    print(f"  d99/D         = {d99_over_D}")
    print(f"  Tc/Tm         = {Tc_over_Tm}")
    print(f"  alpha         = {alpha}°")
    print(f"  L/D           = {L_over_D}")
    
    # Phase 1
    print_section("Phase 1: 기본 열역학")
    thermo = phase1_thermodynamics(Te, Me, Pr)
    print_param("gamma", thermo['gamma'])
    print_param("cp", thermo['cp'], "J/(kg·K)")
    print_param("R", thermo['R'], "J/(kg·K)")
    print_param("a", thermo['a'], "m/s")
    print_param("ue", thermo['ue'], "m/s")
    print_param("mu_e", thermo['mu_e'], "Pa·s")
    
    # Phase 2
    print_section("Phase 2: pe 역산 (Re_D anchor)")
    pressure = phase2_pressure(thermo, Re_D, D)
    print_param("rho_e", pressure['rho_e'], "kg/m³")
    print_param("pe", pressure['pe'], "Pa")
    print_param("nu_e", pressure['nu_e'], "m²/s")
    print_param("Re_D (검증)", pressure['Re_D_check'], fmt=".1f")
    print(f"\n  ★ pe = {pressure['pe']:.0f} Pa ({pressure['pe']/101325:.3f} atm)")
    
    # Phase 3
    print_section("Phase 3: 경계층 파라미터")
    bl = phase3_boundary_layer(thermo, pressure, d99_over_D)
    print_param("d99", bl['d99']*1000, "mm", fmt=".1f")
    print_param("Re_d99", bl['Re_d99'], fmt=".0f")
    
    # Phase 4
    print_section("Phase 4: 벽면 물성")
    wall = phase4_wall_properties(thermo, pressure)
    print_param("Tte", wall['Tte'], "K")
    print_param("pte", wall['pte'], "Pa")
    print_param("Taw", wall['Taw'], "K")
    print_param("rho_w", wall['rho_w'], "kg/m³")
    print_param("mu_w", wall['mu_w'], "Pa·s")
    print_param("nu_w", wall['nu_w'], "m²/s")
    
    # Phase 5
    print_section("Phase 5: PyFR 상수 + 마찰 속도 추정")
    derived = phase5_derived(thermo, pressure, wall, bl)
    print_param("cpTref", derived['cpTref'])
    print_param("cpTs", derived['cpTs'])
    print_param("u_tau (추정)", derived['u_tau_est'], "m/s")
    print_param("Re_tau (추정)", derived['Re_tau_est'], fmt=".0f")
    print_param("Re_theta (추정)", derived['Re_theta_est'], fmt=".0f")
    print_param("d_nu", derived['d_nu'], "m")
    
    # Phase 6
    print_section("Phase 6: Coolant Jet 물성 (Film Cooling)")
    cool = phase6_coolant_jet(thermo, pressure, Tc_over_Tm, BR, D)
    print_param("Tc", cool['Tc'], "K")
    print_param("gamma_c", cool['gamma_c'])
    print_param("rho_c", cool['rho_c'], "kg/m³")
    print_param("DR (ρc/ρe)", cool['DR'], fmt=".3f")
    print_param("Δγ/γ", cool['delta_gamma_pct'], "%", fmt=".2f")

    gamma_status = "INFO" if cool['delta_gamma_pct'] < 2 else "WARNING" if cool['delta_gamma_pct'] < 5 else "ERROR"
    print(f"  Tier 1-F: {gamma_status} (상수 γ 가정 {'합리적' if gamma_status == 'INFO' else '주의'})")

    print(f"\n  BR = {cool['BR']:.2f}")
    print_param("Uc", cool['Uc'], "m/s")
    print_param("VR", cool['VR'], fmt=".4f")
    print_param("I (momentum)", cool['I'], fmt=".4f")
    print_param("Ptc", cool['Ptc'], "Pa")
    print_param("PR (Ptc/pe)", cool['PR'], fmt=".4f")
    print_param("Cd_gritsch", cool['Cd_gritsch'], fmt=".4f")
    print_param("Cd_actual", cool['Cd_actual'], fmt=".4f")
    print(f"  converged      = {cool['converged']} ({cool['n_iter']} iter)")
    
    # Tier 1 검증
    print_section("Tier 1: 자기일관성 검증")
    checks = tier1_validation(thermo, pressure, bl, wall, derived)
    all_pass = True
    for name, err, tol, status in checks:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name:20s}: err={err:.4f}%  (tol={tol}%)  [{status}]")
        if status != "PASS":
            all_pass = False
    print(f"\n  종합: {'ALL PASS ✓' if all_pass else 'FAIL DETECTED ✗'}")
    
    # Wenzel 비교
    print_section("Wenzel 비교 범위 (다양한 d99)")
    wr = wenzel_range_check(thermo, pressure, wall)
    print(f"  {'d99[mm]':>8s} {'d99/D':>6s} {'u_tau':>8s} {'Re_tau':>7s} {'Re_theta':>9s} {'Cf':>9s} {'Wenzel':>12s}")
    print(f"  {'-'*8:>8s} {'-'*6:>6s} {'-'*8:>8s} {'-'*7:>7s} {'-'*9:>9s} {'-'*9:>9s} {'-'*12:>12s}")
    for r in wr:
        print(f"  {r['d99_mm']:8.1f} {r['d99_D']:6.2f} {r['u_tau']:8.2f} {r['Re_tau']:7.0f} {r['Re_theta']:9.0f} {r['cf']:9.5f} {r['wenzel']:>12s}")
    
    # PyFR [constants] 포맷 출력
    print_section("PyFR [constants] 출력")
    print(f";PyFR Constants")
    print(f"gamma  = {thermo['gamma']:.5e}")
    print(f"mu     = {thermo['mu_e']:.5e}")
    print(f"nu     = {pressure['nu_e']:.5e}")
    print(f"Pr     = {Pr:.5e}")
    print(f"cpTref = {derived['cpTref']:.5e}")
    print(f"cpTs   = {derived['cpTs']:.5e}")
    print()
    print(f";User Defined Constants")
    print(f"R      = {thermo['R']:.5e}")
    print(f"ae     = {thermo['a']:.5e}")
    print(f"pe     = {pressure['pe']:.5e}")
    print(f"pte    = {wall['pte']:.5e}")
    print(f"ue     = {thermo['ue']:.5e}")
    print(f"Te     = {Te:.5e}")
    print(f"Tte    = {wall['Tte']:.5e}")
    print(f"cp     = {thermo['cp']:.5e}")
    print(f"rhoe   = {pressure['rho_e']:.5e}")
    print(f"d99_0  = {bl['d99']:.5e}")
    print(f"Me     = {Me:.5e}")
    print(f"Taw    = {wall['Taw']:.5e}")
    print(f"muw    = {wall['mu_w']:.5e}")
    print(f"rhobw  = {wall['rho_w']:.5e}")
    print(f"nuw    = {wall['nu_w']:.5e}")
    
    # 이전 ini 비교
    print_section("이전 ini (pe=81000) 비교")
    Re_D_old = pressure['rho_e'] * thermo['ue'] * D / thermo['mu_e']
    # pe=81000일 때의 Re_D
    rho_old = 81000 / (thermo['R'] * Te)
    Re_D_at_81k = rho_old * thermo['ue'] * D / thermo['mu_e']
    print(f"  현재: pe = {pressure['pe']:.0f} Pa → Re_D = {Re_D:.0f}")
    print(f"  이전: pe = 81,000 Pa → Re_D = {Re_D_at_81k:.0f}")
    print(f"  차이: pe 비율 = {81000/pressure['pe']:.2f}x, Re_D 비율 = {Re_D_at_81k/Re_D:.2f}x")
    
    return {
        'thermo': thermo, 'pressure': pressure, 'bl': bl,
        'wall': wall, 'derived': derived, 'coolant': cool,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Film Cooling CTBL 파라미터 결정')
    parser.add_argument('--Te', type=float, default=537.0, help='Edge temperature [K]')
    parser.add_argument('--Me', type=float, default=0.3, help='Edge Mach number')
    parser.add_argument('--Re_D', type=int, default=13000, help='Target Re_D (anchor)')
    parser.add_argument('--D', type=float, default=0.010, help='Hole diameter [m]')
    parser.add_argument('--d99_over_D', type=float, default=0.5, help='delta99/D ratio')
    parser.add_argument('--Tc_ratio', type=float, default=0.54, help='Tc/Tm ratio')
    
    args = parser.parse_args()
    main(Te=args.Te, Me=args.Me, Re_D=args.Re_D, D=args.D, 
         d99_over_D=args.d99_over_D, Tc_over_Tm=args.Tc_ratio)
