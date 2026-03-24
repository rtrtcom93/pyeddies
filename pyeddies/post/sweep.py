"""sweep.py — Multi-station streamwise sweep + Tier 2 validation.

Processes multiple x-stations at once and generates summary tables.
"""
import os

import numpy as np

from .core import FlowField, load_schlatter_dns
from .profile import extract_wall_normal_profile, WallNormalProfile
from .wall import (compute_wall_properties, compute_wall_properties_lagrange,
                   compute_cf, mu_sutherland)
from .integral import compute_bl_integrals
from .turbstats import reynolds_stress, favre_stress, normalize_reynolds_stress
from .transforms import van_driest_transform


class StreamwiseSweep:
    """Multi-station profile extraction + Tier 2 validation."""

    def __init__(self, flow_field, x_stations, flow_params):
        """
        Parameters
        ----------
        flow_field : FlowField
            Loaded VTU data.
        x_stations : list of float
            Streamwise positions [m].
        flow_params : dict
            Must contain: u_e, rho_e, mu_e, gamma, Me, Te, d99_inlet, R_gas
            Optional: mu0, T0, S (Sutherland params)
        """
        self.ff = flow_field
        self.x_stations = list(x_stations)
        self.fp = flow_params

        # Results storage
        self.profiles = {}    # x -> WallNormalProfile
        self.wall_data = {}   # x -> dict (tau_w, u_tau, Cf, ...)
        self.bl_data = {}     # x -> dict (delta_99, theta, Re_theta, ...)
        self.turb_data = {}   # x -> dict (Reynolds/Favre stresses)

    def compute_all(self, z_mid=None, ndigits=9,
                    use_lagrange=False, mesh_params=None, n_sub=50):
        """Extract profiles and compute all quantities at every station.

        Parameters
        ----------
        z_mid : float or None
            Spanwise midplane for slicing.
        ndigits : int
            Binning precision.
        use_lagrange : bool
            If True, use Lagrange interpolation for sub-element resolution.
            Requires mesh_params.
        mesh_params : dict or None
            Required when use_lagrange=True.
            Keys: 'h_wall', 'growth_rate', 'n_bl_layers'.
        n_sub : int
            Subdivisions per element (only used with use_lagrange=True).
        """
        if use_lagrange:
            from .profile import extract_profile_lagrange
            if mesh_params is None:
                raise ValueError("mesh_params required when use_lagrange=True")

        fp = self.fp
        R_gas = fp['R_gas']
        mu0 = fp.get('mu0', 1.716e-5)
        T0 = fp.get('T0', 273.15)
        S_suth = fp.get('S', 110.4)
        u_e = fp['u_e']
        rho_e = fp['rho_e']
        mu_e = fp['mu_e']
        nu_e = mu_e / rho_e

        for x_st in self.x_stations:
            try:
                if use_lagrange:
                    prof, elements, prof_nodes = extract_profile_lagrange(
                        self.ff, x_st, mesh_params,
                        n_sub=n_sub, z_mid=z_mid,
                        ndigits=ndigits, R_gas=R_gas
                    )
                    self.profiles[x_st] = prof

                    wp = compute_wall_properties_lagrange(
                        prof, elements, prof_nodes,
                        R_gas=R_gas, mu0=mu0, T0=T0, S=S_suth
                    )
                else:
                    prof = extract_wall_normal_profile(
                        self.ff, x_st, z_mid=z_mid,
                        ndigits=ndigits, R_gas=R_gas
                    )
                    self.profiles[x_st] = prof

                    wp = compute_wall_properties(
                        prof, R_gas=R_gas,
                        mu0=mu0, T0=T0, S=S_suth
                    )

                wp['Cf'] = compute_cf(wp['tau_w'], rho_e, u_e)
                self.wall_data[x_st] = wp

                # BL integrals
                rho_arr = prof.rho if prof.has_field('rho') else None
                bl = compute_bl_integrals(
                    wp['y'], wp['U_mean'], rho=rho_arr,
                    u_e=u_e, rho_e=rho_e,
                    nu_e=nu_e, nu_w=wp['nu_w'], u_tau=wp['u_tau']
                )
                self.bl_data[x_st] = bl

                # Turbulence stats (if tavg fields available)
                turb = {}
                if prof.has_field('uu'):
                    turb['reynolds'] = reynolds_stress(prof)
                if prof.has_field('rhouu'):
                    turb['favre'] = favre_stress(prof)
                self.turb_data[x_st] = turb

                print(f"  x={x_st*1e3:.1f}mm: u_tau={wp['u_tau']:.3f}, "
                      f"Cf={wp['Cf']:.5f}, "
                      f"d99={bl['delta_99']*1e3:.2f}mm, "
                      f"Re_tau={bl.get('Re_tau', 0):.0f}, "
                      f"Re_theta={bl.get('Re_theta', 0):.0f}")

            except Exception as e:
                print(f"  x={x_st*1e3:.1f}mm: FAILED — {e}")

    def cf_vs_re_theta(self):
        """Return arrays of (x, Cf, Re_theta) for plotting.

        Returns
        -------
        x_arr, Cf_arr, Re_th_arr : np.ndarray
        """
        x_list, cf_list, reth_list = [], [], []
        for x_st in sorted(self.wall_data.keys()):
            wp = self.wall_data[x_st]
            bl = self.bl_data.get(x_st, {})
            if 'Re_theta' in bl:
                x_list.append(x_st)
                cf_list.append(wp['Cf'])
                reth_list.append(bl['Re_theta'])
        return np.array(x_list), np.array(cf_list), np.array(reth_list)

    def recovery_distance(self, dns_data=None, threshold=0.05):
        """SEM recovery assessment.

        Computes L2 error of Van Driest-transformed u+ vs DNS
        at each station. Recovery = first station where error < threshold.

        Parameters
        ----------
        dns_data : dict or None
            Schlatter DNS data (y_plus, u_plus). Loaded automatically if None.
        threshold : float
            L2 relative error threshold.

        Returns
        -------
        dict: {x_station: L2_error}, recovery_x
        """
        if dns_data is None:
            dns_data = load_schlatter_dns(1410)

        yp_dns = dns_data['y_plus']
        up_dns = dns_data['u_plus']

        errors = {}
        recovery_x = None

        for x_st in sorted(self.wall_data.keys()):
            wp = self.wall_data[x_st]
            yp = wp['y_plus']
            up = wp['u_plus']
            rho = wp['rho_mean']
            rho_w = wp['rho_w']

            # Van Driest transform
            up_vd = van_driest_transform(yp, up, rho, rho_w)

            # Interpolate DNS to simulation y+ grid (log-layer: y+=30~200)
            mask = (yp >= 30) & (yp <= 200)
            if np.sum(mask) < 3:
                continue

            yp_sim = yp[mask]
            up_vd_sim = up_vd[mask]

            # Interpolate DNS to same y+
            up_dns_interp = np.interp(yp_sim, yp_dns, up_dns)

            # L2 relative error
            l2_err = np.sqrt(np.mean((up_vd_sim - up_dns_interp)**2)) / \
                     np.sqrt(np.mean(up_dns_interp**2))

            errors[x_st] = l2_err
            if recovery_x is None and l2_err < threshold:
                recovery_x = x_st

        return errors, recovery_x

    def tier2_summary(self):
        """Print Tier 2 validation summary table."""
        from ..correlations import cf_smits, re_tau_schlatter, van_driest_ii_adiabatic

        fp = self.fp
        gamma = fp['gamma']
        Me = fp['Me']

        print("\n" + "=" * 60)
        print("  Tier 2 Physics Validation Summary")
        print("=" * 60)

        for x_st in sorted(self.wall_data.keys()):
            wp = self.wall_data[x_st]
            bl = self.bl_data.get(x_st, {})
            Re_th = bl.get('Re_theta', None)
            Re_t = bl.get('Re_tau', None)

            print(f"\n--- x = {x_st*1e3:.1f} mm ---")

            # Cf check
            if Re_th is not None:
                Cf_inc = cf_smits(Re_th)
                Cf_comp, Fc = van_driest_ii_adiabatic(Cf_inc, Me, gamma)
                ratio_cf = wp['Cf'] / Cf_comp if Cf_comp > 0 else float('nan')
                status = "PASS" if 0.80 < ratio_cf < 1.20 else "FAIL"
                print(f"  Cf:     sim={wp['Cf']:.5f}, corr={Cf_comp:.5f}, "
                      f"ratio={ratio_cf:.3f} [{status}]")

            # Re_tau check
            if Re_th is not None and Re_t is not None:
                Re_tau_corr = re_tau_schlatter(Re_th)
                ratio_rt = Re_t / Re_tau_corr if Re_tau_corr > 0 else float('nan')
                status = "PASS" if 0.85 < ratio_rt < 1.15 else "FAIL"
                print(f"  Re_tau: sim={Re_t:.0f}, corr={Re_tau_corr:.0f}, "
                      f"ratio={ratio_rt:.3f} [{status}]")

            # Log-law check
            yp = wp['y_plus']
            up = wp['u_plus']
            mask_log = (yp >= 30) & (yp <= 200)
            if np.sum(mask_log) >= 3:
                kappa, B = 0.41, 5.2
                up_theory = (1.0 / kappa) * np.log(yp[mask_log]) + B
                max_err = np.max(np.abs(up[mask_log] - up_theory) / up_theory)
                status = "PASS" if max_err < 0.05 else "FAIL"
                print(f"  Log-law: max_err={max_err:.3f} [{status}]")

            # H12 check
            H12 = bl.get('H12', None)
            if H12 is not None:
                status = "PASS" if 1.3 < H12 < 1.5 else "WARN"
                print(f"  H12:    {H12:.3f} [{status}]")

            # Reynolds stress peak check
            turb = self.turb_data.get(x_st, {})
            rs = turb.get('reynolds', turb.get('favre', {}))
            if 'uu' in rs or 'uu_favre' in rs:
                uu_key = 'uu_favre' if 'uu_favre' in rs else 'uu'
                uu_peak = np.max(rs[uu_key]) / wp['u_tau']**2
                status = "PASS" if 5.0 < uu_peak < 11.0 else "WARN"
                print(f"  <u'u'>/u_tau^2 peak: {uu_peak:.1f} [{status}]")

        print("\n" + "=" * 60)

    def to_csv(self, output_dir):
        """Save all profiles and summary to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Individual profiles
        prof_dir = os.path.join(output_dir, 'profiles')
        os.makedirs(prof_dir, exist_ok=True)
        for x_st, prof in self.profiles.items():
            fname = f"profile_x{x_st*1e3:.0f}mm.csv"
            prof.to_csv(os.path.join(prof_dir, fname))

        # Summary CSV
        rows = []
        for x_st in sorted(self.wall_data.keys()):
            wp = self.wall_data[x_st]
            bl = self.bl_data.get(x_st, {})
            rows.append({
                'x_mm': x_st * 1e3,
                'u_tau': wp['u_tau'],
                'tau_w': wp['tau_w'],
                'Cf': wp.get('Cf', 0),
                'T_w': wp['T_w'],
                'delta_99_mm': bl.get('delta_99', 0) * 1e3,
                'theta_mm': bl.get('theta', 0) * 1e3,
                'delta_star_mm': bl.get('delta_star', 0) * 1e3,
                'H12': bl.get('H12', 0),
                'Re_theta': bl.get('Re_theta', 0),
                'Re_tau': bl.get('Re_tau', 0),
            })

        if rows:
            import csv
            summary_path = os.path.join(output_dir, 'summary.csv')
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Summary saved: {summary_path}")
