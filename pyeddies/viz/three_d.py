"""3D visualization — PyVista (optional dependency)."""


def plot_isosurface(mesh, variable, value, **kwargs):
    """Iso-surface visualization (Q-criterion etc.).

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid or similar
    variable : str
    value : float
    **kwargs : passed to pyvista.Plotter.add_mesh

    Returns
    -------
    pyvista.Plotter
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("3D visualization requires pyvista: pip install pyvista")

    iso = mesh.contour([value], scalars=variable)
    pl = pv.Plotter()
    pl.add_mesh(iso, **kwargs)
    return pl
