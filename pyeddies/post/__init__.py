"""pyeddies.post — Post-processing pipeline for high-order CFD simulations."""

from .core import plane_local_coords, load_schlatter_dns
from .profile import WallNormalProfile
from .wall import (
    mu_sutherland, compute_tau_w, compute_u_tau, compute_cf,
    wall_units, compute_wall_properties, compute_wall_properties_lagrange,
)
from .turbstats import reynolds_stress, favre_stress, turbulence_intensity
from .transforms import van_driest_transform, crocco_busemann, semi_local_scaling
from .integral import (
    delta_99, momentum_thickness, displacement_thickness,
    shape_factor, re_theta, re_tau, compute_bl_integrals,
)


# pyvista-dependent classes: lazy import to avoid crash without pyvista
def __getattr__(name):
    if name == 'FlowField':
        from .core import FlowField
        return FlowField
    if name == 'extract_wall_normal_profile':
        from .profile import extract_wall_normal_profile
        return extract_wall_normal_profile
    if name == 'extract_profile_lagrange':
        from .profile import extract_profile_lagrange
        return extract_profile_lagrange
    if name == 'StreamwiseSweep':
        from .sweep import StreamwiseSweep
        return StreamwiseSweep
    raise AttributeError(f"module 'pyeddies.post' has no attribute {name!r}")
