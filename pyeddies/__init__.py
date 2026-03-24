"""pyeddies — Boundary layer dynamics & turbulence post-processing toolkit."""

__version__ = "0.1.0"


# Lazy imports for pyvista-dependent classes
def __getattr__(name):
    if name == "FlowField":
        from pyeddies.post.core import FlowField
        return FlowField
    if name == "StreamwiseSweep":
        from pyeddies.post.sweep import StreamwiseSweep
        return StreamwiseSweep
    raise AttributeError(f"module 'pyeddies' has no attribute {name!r}")
