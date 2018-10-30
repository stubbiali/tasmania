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
from tasmania.core.physics_composite import DiagnosticComponentComposite, \
											PhysicsComponentComposite, \
											ConcurrentCoupling, \
											ParallelSplitting, \
											SequentialUpdateSplitting
from tasmania.dynamics.diagnostics import HorizontalVelocity, \
										  IsentropicDiagnostics as RawIsentropicDiagnostics, \
										  WaterConstituent
from tasmania.dynamics.homogeneous_isentropic_dycore import HomogeneousIsentropicDynamicalCore
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.dynamics.isentropic_dycore import IsentropicDynamicalCore
from tasmania.dynamics.isentropic_fluxes import HorizontalIsentropicFlux, \
												VerticalIsentropicFlux, \
												HorizontalHomogeneousIsentropicFlux
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics.isentropic_state import get_default_isentropic_state, \
											   get_isothermal_isentropic_state
from tasmania.dynamics.sedimentation_flux import SedimentationFlux
from tasmania.dynamics.vertical_damping import VerticalDamping
from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.grids.topography import Topography1d, Topography2d
from tasmania.physics.coriolis import ConservativeIsentropicCoriolis
from tasmania.physics.isentropic_diagnostics import IsentropicDiagnostics, \
													IsentropicVelocityComponents
from tasmania.physics.isentropic_tendencies import NonconservativeIsentropicPressureGradient, \
												   ConservativeIsentropicPressureGradient, \
												   VerticalIsentropicAdvection, \
												   PrescribedSurfaceHeating
from tasmania.physics.microphysics import Kessler, RaindropFallVelocity, \
										  SaturationAdjustmentKessler
from tasmania.plot.animation import Animation
from tasmania.plot.assemblers import PlotsOverlapper, SubplotsAssembler
from tasmania.plot.contour_xz import make_contour_xz
from tasmania.plot.contourf_xy import make_contourf_xy
from tasmania.plot.contourf_xz import make_contourf_xz
from tasmania.plot.grid import plot_grid_vertical_section, plot_grid_xz, plot_grid_yz
from tasmania.plot.plot_monitors import Plot1d, Plot2d, Plot3d
from tasmania.plot.profile_1d import plot_horizontal_profile, plot_vertical_profile, \
									 plot_vertical_profile_with_respect_to_vertical_height
from tasmania.plot.quiver_xy import make_quiver_xy
from tasmania.plot.topography import plot_topography_3d
from tasmania.utils.data_utils import make_data_array_2d, make_data_array_3d, \
									  make_raw_state, make_state
from tasmania.utils.meteo_utils import get_isothermal_isentropic_analytical_solution
from tasmania.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.utils.storage_utils import load_netcdf_dataset, NetCDFMonitor
from plot.plot_utils import get_figure_and_axes, set_plot_properties


__version__ = '0.2.0'

__all__ = (
	DynamicalCore,
	Model,
	DiagnosticComponentComposite, PhysicsComponentComposite,
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
	Plot1d, Plot2d, Plot3d,
	PlotsOverlapper, SubplotsAssembler, Animation,
	make_contour_xz, make_contourf_xy, make_contourf_xz, make_quiver_xy,
	plot_horizontal_profile, plot_vertical_profile,
	plot_vertical_profile_with_respect_to_vertical_height,
	plot_grid_vertical_section, plot_grid_xz, plot_grid_yz,
	plot_topography_3d,
	make_data_array_2d, make_data_array_3d,
	make_raw_state, make_state,
	get_isothermal_isentropic_analytical_solution,
	ConstantNotFoundError, TimeInconsistencyError,
	load_netcdf_dataset, NetCDFMonitor,
	get_figure_and_axes, set_plot_properties,
)
