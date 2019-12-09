# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
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
# burgers
from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.burgers.physics.diffusion import BurgersHorizontalDiffusion
from tasmania.python.burgers.state import ZhaoSolutionFactory, ZhaoStateFactory

# dwarfs
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.dwarfs.horizontal_hyperdiffusion import HorizontalHyperDiffusion
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping

# framework
from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Stepper,
    TendencyComponent,
)
from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.framework.offline_diagnostics import (
    OfflineDiagnosticComponent,
    RMSD,
    RRMSD,
)
from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.framework.promoters import Diagnostic2Tendency, Tendency2Diagnostic
from tasmania.python.framework.sequential_tendency_splitting import (
    SequentialTendencySplitting,
)
from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)

# grids
from tasmania.python.grids.domain import Domain
from tasmania.python.grids.grid import Grid, PhysicalGrid, NumericalGrid
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids.horizontal_grid import (
    HorizontalGrid,
    PhysicalHorizontalGrid,
    NumericalHorizontalGrid,
)
from tasmania.python.grids.topography import (
    Topography,
    PhysicalTopography,
    NumericalTopography,
)

# isentropic
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore

from tasmania.python.isentropic.physics.coriolis import IsentropicConservativeCoriolis
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
from tasmania.python.isentropic.physics.turbulence import IsentropicSmagorinsky
from tasmania.python.isentropic.physics.vertical_advection import (
    IsentropicVerticalAdvection,
    PrescribedSurfaceHeating,
)
from tasmania.python.isentropic.state import (
    get_isentropic_state_from_brunt_vaisala_frequency,
    get_isentropic_state_from_temperature,
)
from tasmania.python.isentropic.utils import AirPotentialTemperature2Diagnostic, AirPotentialTemperature2Tendency

# physics
from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustment,
    KesslerSedimentation,
)
from tasmania.python.physics.microphysics.old_kessler import (
    KesslerMicrophysics as OldKesslerMicrophysics,
    KesslerSaturationAdjustment as OldKesslerSaturationAdjustment
)
from tasmania.python.physics.microphysics.utils import Clipping, Precipitation
from tasmania.python.physics.turbulence import Smagorinsky2d

# plot
from tasmania.python.plot.animation import Animation
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.monitors import Plot, PlotComposite
from tasmania.python.plot.offline import Line
from tasmania.python.plot.patches import Circle, Rectangle
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
from tasmania.python.utils.storage_utils import (
    deepcopy_array_dict,
    deepcopy_dataarray,
    deepcopy_dataarray_dict,
    get_dataarray_3d,
    get_dataarray_dict,
    get_array_dict,
    get_dataarray_2d,
    zeros,
)
from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.meteo_utils import (
    get_isothermal_isentropic_analytical_solution,
)
from tasmania.python.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.python.utils.io_utils import load_netcdf_dataset, NetCDFMonitor
from tasmania.python.utils.utils import get_time_string


__version__ = "0.6.1"


__all__ = (
    AirPotentialTemperature2Diagnostic,
    AirPotentialTemperature2Tendency,
    Animation,
    BurgersAdvection,
    BurgersDynamicalCore,
    BurgersHorizontalDiffusion,
    BurgersStepper,
    CDF,
    Circle,
    Clipping,
    ConcurrentCoupling,
    ConstantNotFoundError,
    Contour,
    Contourf,
    DataArrayDictOperator,
    Diagnostic2Tendency,
    DiagnosticComponent,
    DiagnosticComponentComposite,
    Domain,
    deepcopy_array_dict,
    deepcopy_dataarray,
    deepcopy_dataarray_dict,
    DynamicalCore,
    get_array_dict,
    get_dataarray_2d,
    get_dataarray_3d,
    get_dataarray_dict,
    get_figure_and_axes,
    get_isentropic_state_from_brunt_vaisala_frequency,
    get_isentropic_state_from_temperature,
    get_isothermal_isentropic_analytical_solution,
    get_time_string,
    Grid,
    HorizontalBoundary,
    HorizontalDiffusion,
    HorizontalGrid,
    HorizontalHyperDiffusion,
    HorizontalSmoothing,
    HorizontalVelocity,
    HovmollerDiagram,
    ImplicitTendencyComponent,
    IsentropicConservativeCoriolis,
    IsentropicDiagnostics,
    IsentropicDynamicalCore,
    IsentropicHorizontalDiffusion,
    IsentropicHorizontalSmoothing,
    IsentropicSmagorinsky,
    IsentropicVelocityComponents,
    IsentropicVerticalAdvection,
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustment,
    KesslerSedimentation,
    Line,
    LineProfile,
    load_netcdf_dataset,
    NetCDFMonitor,
    NumericalGrid,
    NumericalHorizontalGrid,
    NumericalTopography,
    OfflineDiagnosticComponent,
    OldKesslerMicrophysics,
    OldKesslerSaturationAdjustment,
    ParallelSplitting,
    PhysicalGrid,
    PhysicalHorizontalGrid,
    PhysicalTopography,
    Plot,
    PlotComposite,
    Precipitation,
    PrescribedSurfaceHeating,
    Quiver,
    Rectangle,
    RMSD,
    RRMSD,
    SequentialTendencySplitting,
    SequentialUpdateSplitting,
    set_axes_properties,
    set_figure_properties,
    Smagorinsky2d,
    Stepper,
    Tendency2Diagnostic,
    TendencyComponent,
    TimeInconsistencyError,
    TimeSeries,
    Topography,
    VerticalDamping,
    WaterConstituent,
    zeros,
    ZhaoSolutionFactory,
    ZhaoStateFactory,
)
