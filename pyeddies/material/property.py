from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union, Optional, Tuple
import numpy as np

ArrayLike = Union[float, int, Sequence[float], np.ndarray]
R_UNIVERSAL = 8.31446261815324  # J/mol/K

def mu_sutherland(
    T,
    T0  = 273.15,
    mu0 = 1.716e-5,
    S   = 110.4
) -> float:
    """Sutherland viscosity law for air."""
    return mu0 * (T / T0)**1.5 * ((T0 + S) / (T + S))

@dataclass(frozen=True)
class Nasa9Region:
    Tmin: float
    Tmax: float
    coeffs: Sequence[float]

    def eval_poly(self, T: np.ndarray) -> np.ndarray:
        c = list(self.coeffs)
        if len(c) == 9:
            c = c[:7]
        if len(c) != 7:
            raise ValueError("NASA-9 region must have 7 or 9 coefficients.")
        a0, a1, a2, a3, a4, a5, a6 = c
        return (a0*T**-2 + a1*T**-1 + a2 + a3*T + a4*T**2 + a5*T**3 + a6*T**4)


class SpecificHeat:
    SPECIES_DB = {
        "air": 28.97e-3, "o2": 31.998e-3, "n2": 28.0134e-3,
        "co2": 44.0095e-3, "h2": 2.01588e-3, "he": 4.0026e-3, "ar": 39.948e-3,
    }

    def __init__(
        self,
        cp_poly: Optional[Sequence[float]] = None,
        cp_nasa9: Optional[Union[List[Nasa9Region],
                                 Tuple[Sequence[float], Sequence[float], float]]] = None,
        *,
        coeff_mode: str = "cp_over_R",      # 'cp_over_R' | 'cp_molar' | 'cp_mass'  (NASA-9 계수의 의미)
        cp_poly_unit: str = "molar",        # 'molar' | 'mass'  (cp_poly 계수의 단위 기준)
        species: Optional[str] = None,
        molar_mass: Optional[float] = None, # [kg/mol]
        R: float = R_UNIVERSAL,
    ) -> None:
        # --- constants & units
        self.R = float(R)
        if species is not None:
            key = species.lower()
            if key not in self.SPECIES_DB:
                raise ValueError(f"Unknown species '{species}'. Available: {list(self.SPECIES_DB)}")
            molar_mass = self.SPECIES_DB[key]
        self.M = molar_mass  # may be None

        if coeff_mode not in {"cp_over_R", "cp_molar", "cp_mass"}:
            raise ValueError("coeff_mode must be 'cp_over_R', 'cp_molar', or 'cp_mass'.")
        self.coeff_mode = coeff_mode

        if cp_poly_unit not in {"molar", "mass"}:
            raise ValueError("cp_poly_unit must be 'molar' or 'mass'.")
        self._cp_poly_unit = cp_poly_unit

        # cp_poly (다항식) 설정 + 정규화 파라미터 초기화
        self.cp_poly = None if cp_poly is None else tuple(cp_poly)
        self._poly_shift = 0.0
        self._poly_scale = 1.0

        # NASA-9 regions 정규화
        self._regions: Optional[List[Nasa9Region]] = None
        if cp_nasa9 is not None:
            if isinstance(cp_nasa9, list) and cp_nasa9 and isinstance(cp_nasa9[0], Nasa9Region):
                regions = list(cp_nasa9)
            elif isinstance(cp_nasa9, tuple) and len(cp_nasa9) == 3:
                low, high, Tmid = cp_nasa9
                if not (len(low) in (7, 9) and len(high) in (7, 9)):
                    raise ValueError("Back-compat cp_nasa9 expects 7 or 9 coeffs per side.")
                regions = [
                    Nasa9Region(Tmin=-np.inf, Tmax=float(Tmid), coeffs=low),
                    Nasa9Region(Tmin=float(Tmid), Tmax=np.inf,  coeffs=high),
                ]
            else:
                raise TypeError("cp_nasa9 must be List[Nasa9Region] or ([low],[high], T_mid).")
            regions.sort(key=lambda r: (r.Tmin, r.Tmax))
            self._regions = regions

        if (self.cp_poly is None) and (self._regions is None):
            raise ValueError("Provide at least one of cp_poly or cp_nasa9.")

    # ---------- helpers ----------
    def _region_masks(self, T: np.ndarray, regs: List[Nasa9Region]) -> List[np.ndarray]:
        """Left-closed, right-open for all but last; last is closed-closed."""
        masks: List[np.ndarray] = []
        for i, r in enumerate(regs):
            if i < len(regs) - 1:
                m = (T >= r.Tmin) & (T < r.Tmax)
            else:
                m = (T >= r.Tmin) & (T <= r.Tmax)
            masks.append(m)
        return masks

    # ---------- NASA-9 evaluation ----------
    def _cp_molar_from_regions(self, T: np.ndarray, allow_extrapolate: bool = False) -> np.ndarray:
        if self._regions is None:
            raise RuntimeError("cp_nasa9 not set")
        cp_molar = np.empty_like(T, dtype=float)
        covered = np.zeros_like(T, dtype=bool)

        masks = self._region_masks(T, self._regions)
        for mask, reg in zip(masks, self._regions):
            if not np.any(mask):
                continue
            base = reg.eval_poly(T[mask])  # unit depends on coeff_mode
            if self.coeff_mode == "cp_over_R":
                val = self.R * base                     # -> J/mol/K
            elif self.coeff_mode == "cp_molar":
                val = base                               # already J/mol/K
            elif self.coeff_mode == "cp_mass":
                if self.M is None:
                    raise ValueError("molar_mass (or species) is required to convert cp_mass -> cp_molar.")
                val = base * self.M                      # J/kg/K * kg/mol
            else:
                raise RuntimeError("Invalid coeff_mode")
            cp_molar[mask] = val
            covered[mask] = True

        if not np.all(covered):
            if not allow_extrapolate:
                raise ValueError(f"Temperatures outside provided NASA-9 ranges: {T[~covered]}")
            # 상수 연장(구간 끝값)으로 보수적 외삽
            Tmiss = T[~covered]
            leftmost = self._regions[0]
            rightmost = self._regions[-1]
            left_mask = T < leftmost.Tmin
            right_mask = T > rightmost.Tmax
            if np.any(left_mask & ~covered):
                val = leftmost.eval_poly(np.array([leftmost.Tmin]))
                if self.coeff_mode == "cp_over_R": val = self.R * val
                elif self.coeff_mode == "cp_mass":
                    if self.M is None: raise ValueError("molar_mass required.")
                    val = val * self.M
                cp_molar[left_mask & ~covered] = float(val)
                covered[left_mask & ~covered] = True
            if np.any(right_mask & ~covered):
                val = rightmost.eval_poly(np.array([rightmost.Tmax]))
                if self.coeff_mode == "cp_over_R": val = self.R * val
                elif self.coeff_mode == "cp_mass":
                    if self.M is None: raise ValueError("molar_mass required.")
                    val = val * self.M
                cp_molar[right_mask & ~covered] = float(val)
                covered[right_mask & ~covered] = True

        return cp_molar

    def Cp(self, T: ArrayLike, unit: str = "mass", *, allow_extrapolate: bool = False) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if self._regions is None:
            # NASA-9이 없으면 poly로 평가
            return self.Cp_poly(T, unit=unit)
        cp_molar = self._cp_molar_from_regions(T, allow_extrapolate=allow_extrapolate)
        if unit == "molar":
            return cp_molar
        elif unit == "mass":
            if self.M is None:
                raise ValueError("molar_mass (or species) required for mass-basis cp.")
            return cp_molar / self.M
        else:
            raise ValueError("unit must be 'mass' or 'molar'")

    # ---------- polynomial evaluation ----------
    def Cp_poly(self, T: ArrayLike, unit: str = "mass") -> np.ndarray:
        if self.cp_poly is None:
            raise RuntimeError("cp_poly not set")
        T = np.asarray(T, dtype=float)
        x = (T - self._poly_shift) / self._poly_scale  # 정규화 반영

        # Horner (계수: 낮은 차수 -> 높은 차수)
        cp = np.zeros_like(x, dtype=float)
        for coef in reversed(self.cp_poly):
            cp = cp * x + coef

        # cp_poly 기준 단위 → 요청 단위
        if self._cp_poly_unit == "mass" and unit == "mass":
            return cp
        if self._cp_poly_unit == "molar" and unit == "molar":
            return cp
        if self._cp_poly_unit == "molar" and unit == "mass":
            if self.M is None:
                raise ValueError("molar_mass (or species) required for mass-basis cp.")
            return cp / self.M
        if self._cp_poly_unit == "mass" and unit == "molar":
            if self.M is None:
                raise ValueError("molar_mass (or species) required for molar-basis cp.")
            return cp * self.M
        raise ValueError("Invalid cp_poly_unit/unit")

    def set_poly_from_data(
        self,
        T: ArrayLike,
        cp: ArrayLike,
        degree: int = 3,
        *,
        unit: str = "mass",
        normalize: bool = True,
        Tref: float | None = None,
        weights: ArrayLike | None = None,
        rcond: float | None = None
    ) -> dict:
        T = np.asarray(T, dtype=float).ravel()
        cp = np.asarray(cp, dtype=float).ravel()
        if T.size != cp.size:
            raise ValueError("T and cp must have the same length.")
        if degree < 0:
            raise ValueError("degree must be >= 0.")
        if unit not in {"mass","molar"}:
            raise ValueError("unit must be 'mass' or 'molar'")

        # 정규화
        if normalize:
            shift = float(np.median(T) if Tref is None else Tref)
            scale = float(np.ptp(T))
            if scale == 0.0:
                scale = 1.0
        else:
            shift, scale = 0.0, 1.0

        x = (T - shift) / scale
        p_high2low = np.polyfit(x, cp, deg=degree,
                                w=None if weights is None else np.asarray(weights),
                                rcond=rcond)
        coeffs_low2high = p_high2low[::-1]
        cp_fit = np.polyval(p_high2low, x)
        rmse = float(np.sqrt(np.mean((cp_fit - cp)**2)))

        self.cp_poly = tuple(coeffs_low2high)
        self._poly_shift = shift
        self._poly_scale = scale
        self._cp_poly_unit = unit

        return {
            "coeffs_low2high": coeffs_low2high,
            "shift": shift,
            "scale": scale,
            "rmse": rmse,
            "degree": degree,
            "unit": unit
        }

    # ---------- Cv and gamma ----------
    def Cv(self, T: ArrayLike, unit: str = "mass", *, allow_extrapolate: bool = False) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        # cp (molar) 먼저 얻은 뒤 R을 빼서 변환
        if self._regions is not None:
            cp_molar = self._cp_molar_from_regions(T, allow_extrapolate=allow_extrapolate)
        elif self.cp_poly is not None:
            # cp_poly 기준 단위를 molar로 변환
            cpm = self.Cp_poly(T, unit="molar")
            cp_molar = cpm
        else:
            raise RuntimeError("No Cp model available.")

        cv_molar = cp_molar - self.R
        if unit == "molar":
            return cv_molar
        elif unit == "mass":
            if self.M is None:
                raise ValueError("molar_mass (or species) required for mass-basis cv.")
            return cv_molar / self.M
        else:
            raise ValueError("unit must be 'mass' or 'molar'")

    def gamma(self, T: ArrayLike, *, allow_extrapolate: bool = False) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        # gamma는 단위 무관
        if self._regions is not None:
            cp_m = self._cp_molar_from_regions(T, allow_extrapolate=allow_extrapolate)
        elif self.cp_poly is not None:
            cp_m = self.Cp_poly(T, unit="molar")
        else:
            raise RuntimeError("No Cp model available.")
        cv_m = cp_m - self.R
        return cp_m / cv_m

    # ---------- unified callable ----------
    def __call__(
        self,
        T: ArrayLike,
        *,
        unit: str = "mass",
        prefer: str = "auto",  # "poly" | "nasa9" | "auto"
        allow_extrapolate: bool = False
    ) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        has_poly = self.cp_poly is not None
        has_nasa = self._regions is not None

        if not (has_poly or has_nasa):
            raise RuntimeError("No Cp model (poly or NASA-9) is available.")

        if prefer == "poly":
            if has_poly:
                return self.Cp_poly(T, unit=unit)
            if has_nasa:
                return self.Cp(T, unit=unit, allow_extrapolate=allow_extrapolate)
        elif prefer == "nasa9":
            if has_nasa:
                return self.Cp(T, unit=unit, allow_extrapolate=allow_extrapolate)
            if has_poly:
                return self.Cp_poly(T, unit=unit)
        elif prefer == "auto":
            # 기본: NASA-9 우선
            if has_nasa:
                return self.Cp(T, unit=unit, allow_extrapolate=allow_extrapolate)
            if has_poly:
                return self.Cp_poly(T, unit=unit)
        else:
            raise ValueError("prefer must be 'auto', 'poly', or 'nasa9'.")

    def pyfr_constants(self, T_ref, *,
                       P_ref=None, Re_target=None, L_ref=None, Me=None,
                       T0_suth=273.15, S_suth=110.4, Pr=0.71) -> dict:
        """Compute PyFR [constants] section values at a reference state.

        Pressure determination (mutually exclusive):
          - P_ref given directly, OR
          - (Re_target, L_ref, Me) → pe back-calculated from Re = rho*U*L/mu

        Returns dict with: gamma, cp, cv, R_specific, a, pe, rho, mu, nu,
        Pr, cpTref, cpTs, ue, Me, and isentropic quantities.
        """
        if self.M is None:
            raise ValueError("molar_mass (or species) required for pyfr_constants.")

        R_specific = self.R / self.M                        # J/(kg·K)
        T = np.atleast_1d(float(T_ref))
        cp_mass = float(self.Cp(T, unit="mass")[0])         # J/(kg·K)
        gam = float(self.gamma(T)[0])
        cv_mass = cp_mass / gam
        a = float(np.sqrt(gam * R_specific * T_ref))

        mu_e = mu_sutherland(T_ref, T0=T0_suth, S=S_suth)

        # --- Pressure determination ---
        if P_ref is not None and Re_target is not None:
            raise ValueError("Specify P_ref OR (Re_target, L_ref, Me), not both.")

        if P_ref is not None:
            pe = float(P_ref)
            if Me is None:
                raise ValueError("Me is required to compute ue.")
        elif Re_target is not None:
            if L_ref is None or Me is None:
                raise ValueError("Re_target requires L_ref and Me.")
            ue = Me * a
            # Re = rho * ue * L_ref / mu  →  rho = Re * mu / (ue * L_ref)
            rho_e = Re_target * mu_e / (ue * L_ref)
            pe = rho_e * R_specific * T_ref
        else:
            raise ValueError("Must specify either P_ref or Re_target.")

        rho_e = pe / (R_specific * T_ref)
        ue = Me * a
        nu_e = mu_e / rho_e

        # --- Isentropic / wall quantities ---
        Tte = T_ref * (1.0 + (gam - 1.0) / 2.0 * Me**2)
        pte = pe * (Tte / T_ref) ** (gam / (gam - 1.0))
        r_recovery = Pr ** (1.0 / 3.0)
        Taw = T_ref + r_recovery * (Tte - T_ref)

        rho_w = pe / (R_specific * Taw)
        mu_w = mu_sutherland(Taw, T0=T0_suth, S=S_suth)
        nu_w = mu_w / rho_w

        # --- cpTref, cpTs (Sutherland in PyFR expressions) ---
        cpTref = a**2 / (gam - 1.0)
        cpTs = cpTref * S_suth / T0_suth

        return {
            "gamma": gam,
            "cp": cp_mass,
            "cv": cv_mass,
            "R_specific": R_specific,
            "a": a,
            "pe": pe,
            "rho_e": rho_e,
            "ue": ue,
            "Me": Me,
            "mu_e": mu_e,
            "nu_e": nu_e,
            "Pr": Pr,
            "T_ref": T_ref,
            "Tte": Tte,
            "pte": pte,
            "Taw": Taw,
            "rho_w": rho_w,
            "mu_w": mu_w,
            "nu_w": nu_w,
            "cpTref": cpTref,
            "cpTs": cpTs,
        }


# ── NASA-9 Air coefficients (200–6000 K) ────────────────────────
AIR_NASA9_REGIONS = [
    Nasa9Region(200.0, 1000.0,
                [2.898903e6, -5.649626e4, 1.437799e3, -1.653609e0,
                 3.062254e-3, -2.2791387e-6, 6.272365e-10]),
    Nasa9Region(1000.0, 6000.0,
                [6.932484e7, -3.61053e5, 1.476665e3, -6.138349e-2,
                 2.027963e-5, -3.075525e-9, 1.888054e-13]),
]


def get_air_nasa9() -> SpecificHeat:
    """Return a SpecificHeat instance for dry air using NASA-9 thermodynamics."""
    return SpecificHeat(cp_nasa9=AIR_NASA9_REGIONS,
                        coeff_mode="cp_mass", species="air")
