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
from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.burgers.physics.diffusion import BurgersHorizontalDiffusion
from tasmania.python.burgers.state import ZhaoSolutionFactory, ZhaoStateFactory
# dwarfs
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.dwarfs.horizontal_hyperdiffusion import HorizontalHyperDiffusion
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
# framework
from tasmania.python.framework.base_components import \
	DiagnosticComponent, ImplicitTendencyComponent, Stepper, TendencyComponent
from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.framework.offline_diagnostics import \
	OfflineDiagnosticComponent, RMSD, RRMSD
from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.framework.sequential_splitting import SequentialUpdateSplitting
# grids
from tasmania.python.grids.domain import Domain
from tasmania.python.grids.grid import Grid, PhysicalGrid, ComputationalGrid
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids.horizontal_grid import \
	HorizontalGrid, PhysicalHorizontalGrid, ComputationalHorizontalGrid
from tasmania.python.grids.topography import \
	Topography, PhysicalTopography, ComputationalTopography
# isentropic
from tasmania.python.isentropic.dynamics.homogeneous_dycore import \
	HomogeneousIsentropicDynamicalCore
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore
from tasmania.python.isentropic.physics.diagnostics import \
	IsentropicDiagnostics, IsentropicVelocityComponents
from tasmania.python.isentropic.physics.isentropic_horizontal_diffusion import \
	IsentropicHorizontalDiffusion
from tasmania.python.isentropic.physics.isentropic_horizontal_smoothing import \
	IsentropicHorizontalSmoothing
from tasmania.python.isentropic.physics.tendencies import \
	NonconservativeIsentropicPressureGradient, ConservativeIsentropicPressureGradient, \
	VerticalIsentropicAdvection, PrescribedSurfaceHeating
from tasmania.python.isentropic.state import \
	get_default_isentropic_state, get_isothermal_isentropic_state
# physics
from tasmania.python.physics.coriolis import ConservativeIsentropicCoriolis
from tasmania.python.physics.microphysics import \
	Kessler, RaindropFallVelocity, SaturationAdjustmentKessler, Sedimentation
# plot
from tasmania.python.plot.animation import Animation
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.monitors import Plot, PlotComposite
from tasmania.python.plot.patches import Circle, Rectangle
from tasmania.python.plot.plot_utils import \
	get_figure_and_axes, set_axes_properties, set_figure_properties
from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.quiver import Quiver
from tasmania.python.plot.trackers import TimeSeries, HovmollerDiagram
# utilities
from tasmania.python.utils.data_utils import \
	make_dataarray_2d, make_dataarray_3d, make_raw_state, make_state
from tasmania.python.utils.dict_utils import \
	add as dict_add, subtract as dict_subtract, multiply as dict_scale, copy as dict_update
from tasmania.python.utils.meteo_utils import get_isothermal_isentropic_analytical_solution
from tasmania.python.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.python.utils.storage_utils import load_netcdf_dataset, NetCDFMonitor
from tasmania.python.utils.utils import get_time_string


__version__ = '0.3.0'


__all__ = (
	Animation,
	BurgersDynamicalCore,
	BurgersHorizontalDiffusion,
	Circle,
	ComputationalGrid,
	ComputationalHorizontalGrid,
	ComputationalTopography,
	ConcurrentCoupling,
	ConservativeIsentropicCoriolis,
	ConservativeIsentropicPressureGradient,
	ConstantNotFoundError,
	Contour,
	Contourf,
	DiagnosticComponent,
	DiagnosticComponentComposite,
	Domain,
	dict_add,
	dict_scale,
	dict_subtract,
	dict_update,
	DynamicalCore,
	get_default_isentropic_state,
	get_figure_and_axes,
	get_isothermal_isentropic_analytical_solution,
	get_isothermal_isentropic_state,
	get_time_string,
	Grid,
	HomogeneousIsentropicDynamicalCore,
	HorizontalDiffusion,
	HorizontalGrid,
	HorizontalHyperDiffusion,
	HorizontalSmoothing,
	HovmollerDiagram,
	ImplicitTendencyComponent,
	IsentropicDiagnostics,
	IsentropicDynamicalCore,
	IsentropicHorizontalDiffusion,
	IsentropicHorizontalSmoothing,
	IsentropicVelocityComponents,
	Kessler,
	LineProfile,
	load_netcdf_dataset,
	make_dataarray_2d,
	make_dataarray_3d,
	make_raw_state,
	make_state,
	NetCDFMonitor,
	HorizontalBoundary,
	NonconservativeIsentropicPressureGradient,
	OfflineDiagnosticComponent,
	ParallelSplitting,
	PhysicalGrid,
	PhysicalHorizontalGrid,
	PhysicalTopography,
	Plot,
	PlotComposite,
	PrescribedSurfaceHeating,
	Quiver,
	RaindropFallVelocity,
	Rectangle,
	RMSD,
	RRMSD,
	SaturationAdjustmentKessler,
	Sedimentation,
	SequentialUpdateSplitting,
	set_axes_properties,
	set_figure_properties,
	Stepper,
	TendencyComponent,
	TimeInconsistencyError,
	TimeSeries,
	Topography,
	VerticalDamping,
	VerticalIsentropicAdvection,
	ZhaoSolutionFactory,
	ZhaoStateFactory,
)
