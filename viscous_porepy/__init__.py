"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.


isort:skip_file

"""

__version__ = "1.2.6"

# ------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

from viscous_porepy.utils.common_constants import *

from viscous_porepy.utils import error, grid_utils
from viscous_porepy.utils.tangential_normal_projection import TangentialNormalProjection

from viscous_porepy.utils import permutations

from viscous_porepy.geometry import (
    intersections,
    distances,
    constrain_geometry,
    map_geometry,
    geometry_property_checks,
    bounding_box,
)

# Parameters
from viscous_porepy.params.bc import (
    BoundaryCondition,
    BoundaryConditionVectorial,
    face_on_side,
)
from viscous_porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from viscous_porepy.params.data import (
    Parameters,
    initialize_data,
    initialize_default_data,
    set_state,
    set_iterate,
)
from viscous_porepy.params.rock import UnitRock, Shale, SandStone, Granite
from viscous_porepy.params.fluid import Water, UnitFluid

# Grids
from viscous_porepy.grids.grid import Grid
from viscous_porepy.grids.fv_sub_grid import FvSubGrid
from viscous_porepy.grids.mortar_grid import MortarGrid
from viscous_porepy.grids.grid_bucket import GridBucket
from viscous_porepy.grids.structured import CartGrid, TensorGrid
from viscous_porepy.grids.leaf_grid import CartLeafGrid
from viscous_porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from viscous_porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from viscous_porepy.grids.point_grid import PointGrid
from viscous_porepy.grids import match_grids
from viscous_porepy.grids.standard_grids import grid_buckets_2d
from viscous_porepy.grids import grid_extrusion

# Fractures
from viscous_porepy.fracs.fractures_3d import Fracture, EllipticFracture, FractureNetwork3d
from viscous_porepy.fracs.fractures_2d import FractureNetwork2d

# Numerics
from viscous_porepy.numerics.discretization import VoidDiscretization
from viscous_porepy.numerics.interface_laws.elliptic_discretization import (
    EllipticDiscretization,
)

# Control volume, elliptic
from viscous_porepy.numerics.fv import fvutils
from viscous_porepy.numerics.fv.mpsa import Mpsa
from viscous_porepy.numerics.fv.fv_elliptic import (
    FVElliptic,
    EllipticDiscretizationZeroPermeability,
)
from viscous_porepy.numerics.fv.tpfa import Tpfa
from viscous_porepy.numerics.fv.mpfa import Mpfa
from viscous_porepy.numerics.fv.biot import Biot, GradP, DivU, BiotStabilization
from viscous_porepy.numerics.fv.source import ScalarSource

# Virtual elements, elliptic
from viscous_porepy.numerics.vem.dual_elliptic import project_flux
from viscous_porepy.numerics.vem.mvem import MVEM
from viscous_porepy.numerics.vem.mass_matrix import MixedMassMatrix, MixedInvMassMatrix
from viscous_porepy.numerics.vem.vem_source import DualScalarSource

# Finite elements, elliptic
from viscous_porepy.numerics.fem.rt0 import RT0

# Mixed-dimensional discretizations and assemblers
from viscous_porepy.numerics.interface_laws.elliptic_interface_laws import (
    RobinCoupling,
    FluxPressureContinuity,
)

from viscous_porepy.numerics.interface_laws.cell_dof_face_dof_map import CellDofFaceDofMap
from viscous_porepy.numerics.mixed_dim import assembler_filters
from viscous_porepy.numerics.mixed_dim.assembler import Assembler

import viscous_porepy.numerics

# Transport related
from viscous_porepy.numerics.fv.upwind import Upwind
from viscous_porepy.numerics.interface_laws.hyperbolic_interface_laws import UpwindCoupling
from viscous_porepy.numerics.fv.mass_matrix import MassMatrix
from viscous_porepy.numerics.fv.mass_matrix import InvMassMatrix

# Contact mechanics
from viscous_porepy.numerics.interface_laws.contact_mechanics_interface_laws import (
    PrimalContactCoupling,
    DivUCoupling,
    MatrixScalarToForceBalance,
    FractureScalarToForceBalance,
)
from viscous_porepy.numerics.contact_mechanics.contact_conditions import ColoumbContact
from viscous_porepy.numerics.contact_mechanics import contact_conditions

# Related to models and solvers
from viscous_porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from viscous_porepy.numerics.linear_solvers import LinearSolver
from viscous_porepy.models.run_models import run_stationary_model, run_time_dependent_model

from viscous_porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot
from viscous_porepy.models.contact_mechanics_model import ContactMechanics

# Visualization
from viscous_porepy.viz.exporter import Exporter
from viscous_porepy.viz.plot_grid import plot_grid, save_img
from viscous_porepy.viz.fracture_visualization import plot_fractures, plot_wells

# Modules
from viscous_porepy.fracs import utils as frac_utils
from viscous_porepy.fracs import meshing, fracture_importer
from viscous_porepy.grids import coarsening, partition, refinement
import viscous_porepy.utils.derived_discretizations
from viscous_porepy.utils.default_domains import (
    CubeDomain,
    SquareDomain,
    UnitSquareDomain,
    UnitCubeDomain,
)

# ad
import viscous_porepy.ad.forward_mode
import viscous_porepy.ad.functions
import viscous_porepy.ad.utils
