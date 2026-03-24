"""2D slice contour plots — matplotlib."""
import numpy as np


def plot_slice_contour(mesh_slice, variable, params=None, **kwargs):
    """Plot 2D contour from a PyVista slice mesh.

    Parameters
    ----------
    mesh_slice : pyvista.PolyData
    variable : str
    params : dict, optional (for x/D axis label etc.)
    **kwargs : cmap, clim, title, figsize, colorbar, levels

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    points = mesh_slice.points
    values = mesh_slice[variable]

    # 2D projection (drop the thinnest axis)
    extent = points.max(axis=0) - points.min(axis=0)
    drop_axis = np.argmin(extent)
    keep = [i for i in range(3) if i != drop_axis]
    x, y = points[:, keep[0]], points[:, keep[1]]

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 5)))

    # triangulation + contourf
    triang = tri.Triangulation(x, y)
    levels = kwargs.get('levels', 50)
    cmap = kwargs.get('cmap', 'coolwarm')
    clim = kwargs.get('clim', None)

    cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap)
    if clim:
        cf.set_clim(*clim)

    if kwargs.get('colorbar', True):
        plt.colorbar(cf, ax=ax, label=variable)

    # x/D label when params available
    if params and isinstance(params, dict) and 'D' in params:
        D = params['D']
        ax.set_xlabel(f'x/D (D={D*1000:.0f}mm)')

    ax.set_aspect('equal')
    ax.set_title(kwargs.get('title', f'{variable}'))
    return fig, ax
