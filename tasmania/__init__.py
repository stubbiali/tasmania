# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# third-party
from tasmania import third_party

# burgers
from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.burgers.physics.diffusion import (
    BurgersHorizontalDiffusion,
)
from tasmania.python.burgers.state import ZhaoSolutionFactory, ZhaoStateFactory

# domain
from tasmania.python.domain.domain import Domain
from tasmania.python.domain.grid import Grid, PhysicalGrid, NumericalGrid
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.horizontal_grid import (
    HorizontalGrid,
    PhysicalHorizontalGrid,
    NumericalHorizontalGrid,
)
from tasmania.python.domain.topography import (
    Topography,
    PhysicalTopography,
    NumericalTopography,
)

# dwarfs
from tasmania.python.dwarfs.diagnostics import (
    HorizontalVelocity,
    WaterConstituent,
)
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.dwarfs.horizontal_hyperdiffusion import (
    HorizontalHyperDiffusion,
)
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping

# framework
from tasmania.python.framework import tag
from tasmania.python.framework.allocators import as_storage, empty, ones, zeros
from tasmania.python.framework.core_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Stepper,
    TendencyComponent,
)
from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.offline_diagnostics import (
    OfflineDiagnosticComponent,
    RMSD,
    RRMSD,
)
from tasmania.python.framework.options import (
    BackendOptions,
    StorageOptions,
    TimeIntegrationOptions,
)
from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.framework.promoter import (
    FromDiagnosticToTendency,
    FromTendencyToDiagnostic,
)
from tasmania.python.framework.register import factorize, register
from tasmania.python.framework.sequential_tendency_splitting import (
    SequentialTendencySplitting,
)
from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)
from tasmania.python.framework.stencil import (
    StencilCompiler,
    StencilDefinition,
    StencilFactory,
    SubroutineDefinition,
)
from tasmania.python.framework.steppers import (
    SequentialTendencyStepper,
    TendencyStepper,
)

# isentropic
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore
from tasmania.python.isentropic.physics.coriolis import (
    IsentropicConservativeCoriolis,
)
from tasmania.python.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
    IsentropicVelocityComponents,
)
from tasmania.python.isentropic.physics.horizontal_diffusion import (
    IsentropicHorizontalDiffusion,
)
from tasmania.python.isentropic.physics.horizontal_smoothing import (
    IsentropicHorizontalSmoothing,
)
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
    IsentropicImplicitVerticalAdvectionPrognostic,
)
from tasmania.python.isentropic.physics.turbulence import IsentropicSmagorinsky
from tasmania.python.isentropic.physics.vertical_advection import (
    IsentropicVerticalAdvection,
    PrescribedSurfaceHeating,
)
from tasmania.python.isentropic.state import (
    get_isentropic_state_from_brunt_vaisala_frequency,
    get_isentropic_state_from_temperature,
)
from tasmania.python.isentropic.utils import (
    AirPotentialTemperatureToDiagnostic,
    AirPotentialTemperatureToTendency,
)

# physics
from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustmentDiagnostic,
    KesslerSaturationAdjustmentPrognostic,
    KesslerSedimentation,
)
from tasmania.python.physics.microphysics.utils import Clipping, Precipitation
from tasmania.python.physics.turbulence import Smagorinsky2d

# plot
from tasmania.python.plot.animation import Animation
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.monitors import Plot, PlotComposite
from tasmania.python.plot.offline import Line
from tasmania.python.plot.patches import Annotation, Circle, Rectangle, Segment
from tasmania.python.plot.plot_utils import (
    get_figure_and_axes,
    set_axes_properties,
    set_figure_properties,
)
from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.quiver import Quiver
from tasmania.python.plot.spectrals import CDF
from tasmania.python.plot.trackers import TimeSeries, HovmollerDiagram

# utilities
from tasmania.python.utils import typingx
from tasmania.python.utils.dict import DataArrayDictOperator
from tasmania.python.utils.exceptions import (
    ConstantNotFoundError,
    TimeInconsistencyError,
)
from tasmania.python.utils.io import load_netcdf_dataset, NetCDFMonitor
from tasmania.python.utils.meteo import (
    get_isothermal_isentropic_analytical_solution,
)
from tasmania.python.utils.storage import (
    deepcopy_array_dict,
    deepcopy_dataarray,
    deepcopy_dataarray_dict,
    get_dataarray_3d,
    get_dataarray_dict,
    get_array_dict,
    get_dataarray_2d,
)
from tasmania.python.utils.time import Timer, get_time_string
from tasmania.python.utils.utils import feed_module

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


__author__ = "ETH Zurich"
__copyright__ = "ETH Zurich"
__license__ = "GPLv3"


# >>> old storage
from gt4py.storage import prepare_numpy

prepare_numpy()
# <<< new storage
# <<<
