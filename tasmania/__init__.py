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
from tasmania.python.core.dycore import DynamicalCore
from tasmania.python.core.model import Model
from tasmania.python.core.offline_diagnostics import OfflineDiagnosticComponent, RMSD, RRMSD
from tasmania.python.core.composite import DiagnosticComponentComposite
from tasmania.python.core.concurrent_coupling import ConcurrentCoupling
from tasmania.python.core.parallel_splitting import ParallelSplitting
from tasmania.python.core.sequential_splitting import SequentialUpdateSplitting
from tasmania.python.dynamics.diagnostics import \
	HorizontalVelocity, \
	IsentropicDiagnostics as RawIsentropicDiagnostics, \
	WaterConstituent
from tasmania.python.dynamics.homogeneous_isentropic_dycore import \
	HomogeneousIsentropicDynamicalCore
from tasmania.python.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.python.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dynamics.isentropic_dycore import IsentropicDynamicalCore
from tasmania.python.dynamics.isentropic_fluxes import \
	HorizontalIsentropicFlux, VerticalIsentropicFlux, HorizontalHomogeneousIsentropicFlux
from tasmania.python.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.python.dynamics.isentropic_state import \
	get_default_isentropic_state, get_isothermal_isentropic_state
from tasmania.python.dynamics.vertical_damping import VerticalDamping
from tasmania.python.grids.grid_xy import GridXY
from tasmania.python.grids.grid_xyz import GridXYZ
from tasmania.python.grids.grid_xz import GridXZ
from tasmania.python.grids.topography import Topography1d, Topography2d
from tasmania.python.physics.coriolis import ConservativeIsentropicCoriolis
from tasmania.python.physics.isentropic_diagnostics import \
	IsentropicDiagnostics, IsentropicVelocityComponents
from tasmania.python.physics.isentropic_tendencies import \
	NonconservativeIsentropicPressureGradient, ConservativeIsentropicPressureGradient, \
	VerticalIsentropicAdvection, PrescribedSurfaceHeating
from tasmania.python.physics.microphysics import \
	Kessler, RaindropFallVelocity, SaturationAdjustmentKessler, Sedimentation
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
from tasmania.python.utils.data_utils import \
	make_data_array_2d, make_data_array_3d, make_raw_state, make_state
from tasmania.python.utils.dict_utils import \
	add as dict_add, subtract as dict_subtract, multiply as dict_scale, copy as dict_update
from tasmania.python.utils.meteo_utils import get_isothermal_isentropic_analytical_solution
from tasmania.python.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.python.utils.storage_utils import load_netcdf_dataset, NetCDFMonitor
from tasmania.python.utils.utils import get_time_string


__version__ = '0.2.0'


__all__ = (
	DynamicalCore,
	Model,
	OfflineDiagnosticComponent, RMSD, RRMSD,
	DiagnosticComponentComposite, ConcurrentCoupling, 
	ParallelSplitting, SequentialUpdateSplitting,
	HorizontalBoundary, HorizontalSmoothing, VerticalDamping,
	HorizontalIsentropicFlux, VerticalIsentropicFlux,
	HorizontalHomogeneousIsentropicFlux,
	IsentropicPrognostic, RawIsentropicDiagnostics,
	IsentropicDynamicalCore, HomogeneousIsentropicDynamicalCore,
	get_default_isentropic_state, get_isothermal_isentropic_state,
	HorizontalVelocity, WaterConstituent,
	GridXY, GridXZ, GridXYZ,
	Topography1d, Topography2d,
	ConservativeIsentropicCoriolis,
	IsentropicDiagnostics, IsentropicVelocityComponents,
	ConservativeIsentropicPressureGradient, NonconservativeIsentropicPressureGradient,
	PrescribedSurfaceHeating, VerticalIsentropicAdvection,
	Kessler, RaindropFallVelocity, SaturationAdjustmentKessler,
	Plot, PlotComposite, Animation,
	LineProfile, TimeSeries, HovmollerDiagram,
	Contour, Contourf, Quiver,
	Circle, Rectangle,
	get_figure_and_axes, set_axes_properties, set_figure_properties,
	make_data_array_2d, make_data_array_3d,
	make_raw_state, make_state,
	dict_add, dict_subtract, dict_scale, dict_update,
	get_isothermal_isentropic_analytical_solution,
	ConstantNotFoundError, TimeInconsistencyError,
	load_netcdf_dataset, NetCDFMonitor,
	get_time_string,
)
