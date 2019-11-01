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
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.profile import LineProfile
from tasmania.python.utils.io_utils import load_netcdf_dataset
from tasmania.python.utils.storage_utils import zeros


@pytest.fixture(scope="module")
def isentropic_dry_data():
    return load_netcdf_dataset("baseline_datasets/isentropic_dry.nc")


@pytest.fixture(scope="module")
def isentropic_moist_data():
    return load_netcdf_dataset("baseline_datasets/isentropic_moist.nc")


@pytest.fixture(scope="module")
def isentropic_moist_sedimentation_data():
    return load_netcdf_dataset("baseline_datasets/isentropic_moist_sedimentation.nc")


@pytest.fixture(scope="module")
def isentropic_moist_sedimentation_evaporation_data():
    return load_netcdf_dataset(
        "baseline_datasets/isentropic_moist_sedimentation_evaporation.nc"
    )


@pytest.fixture(scope="module")
def validation_data():
    return load_netcdf_dataset("baseline_datasets/validation.nc")


@pytest.fixture(scope="module")
def physical_constants():
    return {
        "air_pressure_at_sea_level": DataArray(1e5, attrs={"units": "Pa"}),
        "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
        "gravitational_acceleration": DataArray(9.81, attrs={"units": "m s^-2"}),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }


@pytest.fixture(scope="module")
def drawer_topography_1d():
    def _drawer_topography_1d(
        grid, topography_units="m", x=None, y=None, axis_name=None, axis_units=None
    ):
        properties = {"linecolor": "black", "linewidth": 1.3}
        return LineProfile(
            grid,
            "topography",
            topography_units,
            x=x,
            y=y,
            z=-1,
            axis_name=axis_name,
            axis_units=axis_units,
            properties=properties,
        )

    return _drawer_topography_1d


@pytest.fixture(scope="module")
def drawer_topography_2d():
    def _drawer_topography_2d(
        grid,
        topography_units="m",
        xaxis_name=None,
        xaxis_units=None,
        yaxis_name=None,
        yaxis_units=None,
    ):
        properties = {"colors": "darkgray", "draw_vertical_levels": False}
        return Contour(
            grid,
            "topography",
            topography_units,
            z=-1,
            xaxis_name=xaxis_name,
            xaxis_units=xaxis_units,
            yaxis_name=yaxis_name,
            yaxis_units=yaxis_units,
            properties=properties,
        )

    return _drawer_topography_2d


class FakeTendencyComponent1(TendencyComponent):
    def __init__(self, domain, grid_type):
        super().__init__(domain, grid_type)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "x_velocity_at_u_locations": {"dims": dims_x, "units": "km hr^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "x_velocity_at_u_locations": {"dims": dims_x, "units": "m s^-2"},
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {"fake_variable": {"dims": dims, "units": "kg m^-2 K^-1"}}

        return return_dict

    def array_call(self, state):
        s = state["air_isentropic_density"]
        su = state["x_momentum_isentropic"]
        u = state["x_velocity_at_u_locations"]

        tendencies = {
            "air_isentropic_density": 1e-3 * s,
            "x_momentum_isentropic": 300 * su,
            "x_velocity_at_u_locations": 50 * u / 3.6,
        }

        diagnostics = {"fake_variable": 2 * s}

        return tendencies, diagnostics


@pytest.fixture(scope="module")
def make_fake_tendency_component_1():
    def _make_fake_tendency_component_1(domain, grid_type):
        return FakeTendencyComponent1(domain, grid_type)

    return _make_fake_tendency_component_1


class FakeTendencyComponent2(TendencyComponent):
    def __init__(self, domain, grid_type):
        super().__init__(domain, grid_type)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg km^-2 K^-1"},
            "fake_variable": {"dims": dims, "units": "kg m^-2 K^-1"},
            "y_velocity_at_v_locations": {"dims": dims_y, "units": "km hr^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        s = state["air_isentropic_density"]
        f = state["fake_variable"]
        v = state["y_velocity_at_v_locations"]

        g = self.grid

        if s.shape == (g.nx, g.ny, g.nz):
            sv = 0.5 * 1e-6 * s * (v[:, :-1, :] / 3.6 + v[:, 1:, :] / 3.6)
        elif s.shape == (g.nx + 1, g.ny + 1, g.nz + 1):
            try:
                backend = s.backend
                dtype = s.dtype
                default_origin = s.default_origin
                sv = zeros(s.shape, backend, dtype, default_origin)
            except AttributeError:
                sv = np.zeros_like(s, dtype=s.dtype)
                
            sv[:, :-1] = 0.5 * 1e-6 * s[:, :-1] * (v[:, :-1] / 3.6 + v[:, 1:] / 3.6)
        else:
            raise RuntimeError()

        tendencies = {"air_isentropic_density": f / 100, "y_momentum_isentropic": sv}

        diagnostics = {}

        return tendencies, diagnostics


@pytest.fixture(scope="module")
def make_fake_tendency_component_2():
    def _make_fake_tendency_component_2(domain, grid_type):
        return FakeTendencyComponent2(domain, grid_type)

    return _make_fake_tendency_component_2
