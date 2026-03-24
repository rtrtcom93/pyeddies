"""FlowParams: params.yaml + material module로 전체 유동 파라미터 자동 산출."""
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class FlowParams:
    """유동 파라미터 통합 관리.

    params.yaml에서 최소 입력만 받고, 나머지는 자동 계산.
    모든 후처리 함수에 이 객체를 전달.

    Usage
    -----
    >>> fp = FlowParams.from_yaml("cases/fc_sr_br1p0_ma03_p4/params.yaml")
    >>> fp.rhoe    # 자동 계산됨
    >>> fp.utau_0  # tune_utau 자동 실행
    >>> sweep = StreamwiseSweep(ff, x_stations, fp)
    """
    # --- YAML에서 읽는 입력값 ---
    case_name: str = ""
    case_type: str = "ctbl"       # "ctbl" or "fc"

    # Flow
    Te: float = 537.0             # [K]
    Me: float = 0.3
    Pr: float = 0.71

    # Reynolds
    Re_D: int = 26000
    D: float = 0.010              # [m]

    # Boundary layer
    d99_inlet: float = 0.0042     # [m]
    kappa: float = 0.41
    B: float = 5.2
    Pi_wake: float = 0.45

    # Domain
    Lx: float = 0.1
    Ly: float = 0.04
    Lz: float = 0.03

    # FC-specific (optional)
    Tc: Optional[float] = None
    BR: Optional[float] = None
    Ptc: Optional[float] = None

    # --- 자동 계산 (post_init) ---
    gamma: float = field(init=False, default=0.0)
    cp: float = field(init=False, default=0.0)
    R_gas: float = field(init=False, default=0.0)
    ae: float = field(init=False, default=0.0)
    ue: float = field(init=False, default=0.0)
    mu_e: float = field(init=False, default=0.0)
    rhoe: float = field(init=False, default=0.0)
    pe: float = field(init=False, default=0.0)
    nu_e: float = field(init=False, default=0.0)
    Tte: float = field(init=False, default=0.0)
    Taw: float = field(init=False, default=0.0)
    rho_w: float = field(init=False, default=0.0)
    mu_w: float = field(init=False, default=0.0)
    nu_w: float = field(init=False, default=0.0)
    utau_0: float = field(init=False, default=0.0)
    dnu_0: float = field(init=False, default=0.0)

    # FC derived
    rhoc: Optional[float] = field(init=False, default=None)
    DR: Optional[float] = field(init=False, default=None)
    VR: Optional[float] = field(init=False, default=None)
    Uc: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        """params.yaml 입력에서 전체 파라미터 체인 자동 계산."""
        from .material.property import get_air_nasa9, mu_sutherland
        from .profile.mean import tune_utau_for_delta99

        air = get_air_nasa9()

        # Phase 1: Thermodynamics
        self.cp = float(air.Cp(self.Te))
        self.gamma = float(air.gamma(self.Te))
        self.R_gas = self.cp * (self.gamma - 1) / self.gamma
        self.ae = float(np.sqrt(self.gamma * self.R_gas * self.Te))
        self.ue = self.Me * self.ae
        self.mu_e = mu_sutherland(self.Te)

        # Phase 2: Pressure from Re_D
        self.rhoe = self.Re_D * self.mu_e / (self.ue * self.D)
        self.pe = self.rhoe * self.R_gas * self.Te
        self.nu_e = self.mu_e / self.rhoe

        # Phase 3: Wall properties
        rf = self.Pr ** (1.0 / 3.0)
        self.Tte = self.Te * (1 + (self.gamma - 1) / 2 * self.Me ** 2)
        self.Taw = self.Te * (1 + rf * (self.gamma - 1) / 2 * self.Me ** 2)
        self.rho_w = self.pe / (self.R_gas * self.Taw)
        self.mu_w = mu_sutherland(self.Taw)
        self.nu_w = self.mu_w / self.rho_w

        # Phase 4: Friction velocity
        self.utau_0 = tune_utau_for_delta99(
            target_delta99=self.d99_inlet,
            ue=self.ue,
            nuw=self.nu_w,
            kappa=self.kappa,
            B=self.B,
            Pi=self.Pi_wake,
        )
        self.dnu_0 = self.nu_w / self.utau_0

        # FC-specific
        if self.Tc is not None:
            self.rhoc = self.pe / (self.R_gas * self.Tc)
            self.DR = self.rhoc / self.rhoe
            if self.BR is not None:
                self.VR = self.BR / self.DR
                self.Uc = self.BR * self.rhoe * self.ue / self.rhoc

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FlowParams':
        """params.yaml 파일에서 FlowParams 생성.

        Parameters
        ----------
        yaml_path : str
            Path to params.yaml

        Returns
        -------
        FlowParams with all derived quantities computed
        """
        with open(yaml_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        kwargs = {}

        # case
        if 'case' in cfg:
            kwargs['case_name'] = cfg['case'].get('name', '')
            kwargs['case_type'] = cfg['case'].get('type', 'ctbl')

        # flow
        if 'flow' in cfg:
            for k in ('Te', 'Me', 'Pr'):
                if k in cfg['flow']:
                    kwargs[k] = cfg['flow'][k]

        # reynolds
        if 'reynolds' in cfg:
            for k in ('Re_D', 'D'):
                if k in cfg['reynolds']:
                    kwargs[k] = cfg['reynolds'][k]

        # boundary_layer
        if 'boundary_layer' in cfg:
            bl = cfg['boundary_layer']
            if 'd99_inlet' in bl:
                kwargs['d99_inlet'] = bl['d99_inlet']
            if 'kappa' in bl:
                kwargs['kappa'] = bl['kappa']
            if 'B' in bl:
                kwargs['B'] = bl['B']
            if 'Pi_wake' in bl:
                kwargs['Pi_wake'] = bl['Pi_wake']

        # domain
        if 'domain' in cfg:
            for k in ('Lx', 'Ly', 'Lz'):
                if k in cfg['domain']:
                    kwargs[k] = cfg['domain'][k]

        # FC-specific — check both 'coolant' and 'filmcool' keys
        for section in ('coolant', 'filmcool'):
            if section in cfg:
                c = cfg[section]
                if 'Tc' in c:
                    kwargs['Tc'] = c['Tc']
                # BR can be 'BR' or 'BR_target'
                if 'BR' in c:
                    kwargs['BR'] = c['BR']
                elif 'BR_target' in c:
                    kwargs['BR'] = c['BR_target']
                if 'Ptc' in c:
                    kwargs['Ptc'] = c['Ptc']
                # D from filmcool overrides reynolds D (hole diameter)
                if 'D' in c:
                    kwargs['D'] = c['D']

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """전체 파라미터를 dict로 반환 (sweep, profile 등에 전달용).

        기존 flow_params dict와 호환.
        """
        d = {
            'Te': self.Te, 'Me': self.Me, 'Pr': self.Pr,
            'gamma': self.gamma, 'cp': self.cp, 'R_gas': self.R_gas,
            'ae': self.ae, 'ue': self.ue, 'mu_e': self.mu_e,
            'rhoe': self.rhoe, 'pe': self.pe, 'nu_e': self.nu_e,
            'rho_e': self.rhoe, 'nu_w': self.nu_w,  # aliases for compat
            'Tte': self.Tte, 'Taw': self.Taw,
            'rho_w': self.rho_w, 'mu_w': self.mu_w,
            'utau_0': self.utau_0, 'dnu_0': self.dnu_0,
            'utau_inlet': self.utau_0,  # alias
            'd99_0': self.d99_inlet, 'd99_inlet': self.d99_inlet,
            'D': self.D, 'Lx': self.Lx, 'Ly': self.Ly, 'Lz': self.Lz,
            'kappa': self.kappa, 'B': self.B, 'Pi_wake': self.Pi_wake,
        }
        if self.Tc is not None:
            d.update({
                'Tc': self.Tc, 'rhoc': self.rhoc, 'DR': self.DR,
                'BR': self.BR, 'VR': self.VR, 'Uc': self.Uc,
                'Ptc': self.Ptc, 'rho_c': self.rhoc,  # alias
            })
        return d

    def __repr__(self):
        lines = [f"FlowParams(case={self.case_name!r}, type={self.case_type!r})"]
        lines.append(f"  Te={self.Te} K, Me={self.Me}, Re_D={self.Re_D}")
        lines.append(f"  pe={self.pe:.0f} Pa, rhoe={self.rhoe:.4f} kg/m³")
        lines.append(f"  ue={self.ue:.3f} m/s, utau={self.utau_0:.3f} m/s")
        if self.Tc is not None:
            lines.append(f"  Tc={self.Tc} K, DR={self.DR:.3f}, BR={self.BR}")
        return "\n".join(lines)
