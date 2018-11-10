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
from tasmania.core.dycore import DynamicalCore
from tasmania.core.model import Model
from tasmania.core.offline_diagnostics import OfflineDiagnosticComponent, RMSD, RRMSD
from tasmania.core.physics_composite import \
	DiagnosticComponentComposite, PhysicsComponentComposite, \
	ConcurrentCoupling, ParallelSplitting, SequentialUpdateSplitting
from tasmania.dynamics.diagnostics import \
	HorizontalVelocity, IsentropicDiagnostics as RawIsentropicDiagnostics, WaterConstituent
from tasmania.dynamics.homogeneous_isentropic_dycore import HomogeneousIsentropicDynamicalCore
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.dynamics.isentropic_dycore import IsentropicDynamicalCore
from tasmania.dynamics.isentropic_fluxes import \
	HorizontalIsentropicFlux, VerticalIsentropicFlux, \
	HorizontalHomogeneousIsentropicFlux
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics.isentropic_state import \
	get_default_isentropic_state, get_isothermal_isentropic_state
from tasmania.dynamics.sedimentation_flux import SedimentationFlux
from tasmania.dynamics.vertical_damping import VerticalDamping
from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.grids.topography import Topography1d, Topography2d
from tasmania.physics.coriolis import ConservativeIsentropicCoriolis
from tasmania.physics.isentropic_diagnostics import \
	IsentropicDiagnostics, IsentropicVelocityComponents
from tasmania.physics.isentropic_tendencies import \
	NonconservativeIsentropicPressureGradient, ConservativeIsentropicPressureGradient, \
	VerticalIsentropicAdvection, PrescribedSurfaceHeating
from tasmania.physics.microphysics import \
	Kessler, RaindropFallVelocity, SaturationAdjustmentKessler
from tasmania.plot.animation import Animation
from tasmania.plot.contour import Contour
from tasmania.plot.contourf import Contourf
from tasmania.plot.monitors import Plot, PlotComposite
from tasmania.plot.plot_utils import \
	get_figure_and_axes, set_axes_properties, set_figure_properties
from tasmania.plot.profile import LineProfile
from tasmania.plot.quiver import Quiver
from tasmania.plot.trackers import TimeSeries, HovmollerDiagram
from tasmania.utils.data_utils import \
	make_data_array_2d, make_data_array_3d, make_raw_state, make_state
from tasmania.utils.meteo_utils import get_isothermal_isentropic_analytical_solution
from tasmania.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.utils.storage_utils import load_netcdf_dataset, NetCDFMonitor


__version__ = '0.2.0'

__all__ = (
	DynamicalCore,
	Model,
	DiagnosticComponentComposite, PhysicsComponentComposite,
	OfflineDiagnosticComponent, RMSD, RRMSD,
	ConcurrentCoupling, ParallelSplitting, SequentialUpdateSplitting,
	HorizontalBoundary, HorizontalSmoothing, VerticalDamping,
	HorizontalIsentropicFlux, VerticalIsentropicFlux,
	HorizontalHomogeneousIsentropicFlux,
	IsentropicPrognostic, RawIsentropicDiagnostics,
	IsentropicDynamicalCore, HomogeneousIsentropicDynamicalCore,
	get_default_isentropic_state, get_isothermal_isentropic_state,
	SedimentationFlux,
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
	get_figure_and_axes, set_axes_properties, set_figure_properties,
	make_data_array_2d, make_data_array_3d,
	make_raw_state, make_state,
	get_isothermal_isentropic_analytical_solution,
	ConstantNotFoundError, TimeInconsistencyError,
	load_netcdf_dataset, NetCDFMonitor,
)
