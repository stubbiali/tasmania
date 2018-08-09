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
from tasmania.dynamics.diagnostics import HorizontalVelocity, IsentropicDiagnostics, \
										  WaterConstituent
from tasmania.dynamics.dycore import DynamicalCore
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.dynamics.isentropic_dycore import IsentropicDynamicalCore
from tasmania.dynamics.isentropic_fluxes import IsentropicHorizontalFlux, \
												IsentropicVerticalFlux
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics.sedimentation_flux import SedimentationFlux
from tasmania.dynamics.vertical_damping import VerticalDamping
from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.grids.topography import Topography1d, Topography2d
from tasmania.model import Model
from tasmania.physics.composite import PhysicsComponentComposite
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
from tasmania.utils.data_utils import make_data_array_2d, make_data_array_3d
from tasmania.utils.exceptions import ConstantNotFoundError, TimeInconsistencyError
from tasmania.utils.storage_utils import load_netcdf_dataset, NetCDFMonitor


__version__ = '0.2.0'

__all__ = (
	DynamicalCore,
	HorizontalBoundary, HorizontalSmoothing, VerticalDamping,
	IsentropicHorizontalFlux, IsentropicVerticalFlux,
	IsentropicPrognostic, IsentropicDiagnostics, IsentropicDynamicalCore,
	SedimentationFlux,
	HorizontalVelocity, WaterConstituent,
	GridXY, GridXZ, GridXYZ,
	Topography1d, Topography2d,
	PhysicsComponentComposite,
	Kessler, RaindropFallVelocity, SaturationAdjustmentKessler,
	Model,
	Plot1d, Plot2d, Plot3d,
	PlotsOverlapper, SubplotsAssembler, Animation,
	make_contour_xz, make_contourf_xy, make_contourf_xz, make_quiver_xy,
	plot_horizontal_profile, plot_vertical_profile,
	plot_vertical_profile_with_respect_to_vertical_height,
	plot_grid_vertical_section, plot_grid_xz, plot_grid_yz,
	plot_topography_3d,
	make_data_array_2d, make_data_array_3d,
	ConstantNotFoundError, TimeInconsistencyError,
	load_netcdf_dataset, NetCDFMonitor,
)
