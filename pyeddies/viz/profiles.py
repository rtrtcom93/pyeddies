"""프로파일 계열 2D 플롯 — matplotlib."""


def plot_uplus_yplus(y_plus, u_plus, ref_data=None, **kwargs):
    """u+(y+) semi-log plot.

    Parameters
    ----------
    y_plus, u_plus : array
    ref_data : dict, optional
        {'dns': (yp, up), 'log_law': (yp, up)} reference data

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
    ax.semilogx(y_plus, u_plus, 'k-', lw=1.5, label='LES')

    if ref_data:
        for name, (yp, up) in ref_data.items():
            ax.semilogx(yp, up, '--', label=name)

    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    ax.legend()
    return fig, ax


def plot_eta_xD(x_over_D, eta_lat, eta_cl=None, ref_data=None, **kwargs):
    """eta(x/D) plot — laterally averaged + centerline.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 5)))
    ax.plot(x_over_D, eta_lat, 'k-o', ms=4, label=r'$\bar{\eta}_{lat}$')

    if eta_cl is not None:
        ax.plot(x_over_D, eta_cl, 'b-s', ms=4, label=r'$\eta_{cl}$')

    if ref_data:
        for name, (xd, eta) in ref_data.items():
            ax.plot(xd, eta, '--', label=name)

    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$\eta$')
    ax.set_ylim(0, None)
    ax.legend()
    return fig, ax


def plot_cf_x(x_stations, cf_values, ref_corr=None, **kwargs):
    """Cf(x) plot. ref_corr: Smits correlation etc.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))
    ax.plot(x_stations, cf_values, 'k-o', ms=4, label='LES')

    if ref_corr:
        for name, (x, cf) in ref_corr.items():
            ax.plot(x, cf, '--', label=name)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$C_f$')
    ax.legend()
    return fig, ax


def plot_reynolds_stress(y_plus, R_ij, component='uu', **kwargs):
    """Reynolds stress profile.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
    ax.plot(y_plus, R_ij, 'k-', lw=1.5, label=component)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(f'$R_{{{component}}}$')
    ax.legend()
    return fig, ax


def plot_psd(f, Pxx, St=None, **kwargs):
    """Power Spectral Density — log-log.

    Parameters
    ----------
    f : array — frequencies
    Pxx : array — PSD values
    St : array, optional — Strouhal numbers (uses St axis if provided)

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))
    x_data = St if St is not None else f
    ax.loglog(x_data, Pxx, 'k-', lw=1.0)
    ax.set_xlabel(r'$St$' if St is not None else r'$f$ [Hz]')
    ax.set_ylabel(r'$P_{xx}$')

    # -5/3 slope reference
    if kwargs.get('slope_ref', False):
        import numpy as np
        x_mid = np.sqrt(x_data[len(x_data)//4] * x_data[3*len(x_data)//4])
        y_mid = Pxx[len(Pxx)//2]
        x_ref = np.array([x_mid/3, x_mid*3])
        y_ref = y_mid * (x_ref/x_mid)**(-5/3)
        ax.loglog(x_ref, y_ref, 'r--', lw=0.8, label=r'$-5/3$')
        ax.legend()

    return fig, ax
