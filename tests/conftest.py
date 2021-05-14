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

from gt4py import gtscript

from tasmania import TendencyComponent
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.profile import LineProfile
from tasmania.python.utils.io import load_netcdf_dataset


@pytest.fixture(scope="module")
def isentropic_data():
    return load_netcdf_dataset("baseline_datasets/isentropic.nc")


@pytest.fixture(scope="module")
def physical_constants():
    return {
        "air_pressure_at_sea_level": DataArray(1e5, attrs={"units": "Pa"}),
        "gas_constant_of_dry_air": DataArray(
            287.05, attrs={"units": "J K^-1 kg^-1"}
        ),
        "gravitational_acceleration": DataArray(
            9.81, attrs={"units": "m s^-2"}
        ),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }


@pytest.fixture(scope="module")
def drawer_topography_1d():
    def _drawer_topography_1d(
        grid,
        topography_units="m",
        x=None,
        y=None,
        axis_name=None,
        axis_units=None,
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
    def __init__(
        self,
        domain,
        grid_type,
        *,
        backend,
        backend_options,
        storage_shape,
        storage_options
    ):
        super().__init__(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.stencil = self.compile("fake1")

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "x_velocity_at_u_locations": {"dims": dims_x, "units": "m s^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "x_velocity": {"dims": dims, "units": "m s^-2"},
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "fake_variable": {"dims": dims, "units": "kg m^-2 K^-1"}
        }

        return return_dict

    def array_call(
        self, state, out_tendencies, out_diagnostics, overwrite_tendencies
    ):
        self.stencil(
            s=state["air_isentropic_density"],
            su=state["x_momentum_isentropic"],
            u=state["x_velocity_at_u_locations"],
            tnd_s=out_tendencies["air_isentropic_density"],
            tnd_su=out_tendencies["x_momentum_isentropic"],
            tnd_u=out_tendencies["x_velocity"],
            fake=out_diagnostics["fake_variable"],
            ow_tnd_s=overwrite_tendencies["air_isentropic_density"],
            ow_tnd_su=overwrite_tendencies["x_momentum_isentropic"],
            ow_tnd_u=overwrite_tendencies["x_velocity"],
            origin=(0, 0, 0),
            domain=(self.grid.nx, self.grid.ny, self.grid.nz),
        )

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="fake1")
    def stencil_numpy(
        s,
        su,
        u,
        tnd_s,
        tnd_su,
        tnd_u,
        fake,
        *,
        ow_tnd_s,
        ow_tnd_su,
        ow_tnd_u,
        origin,
        domain
    ):
        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        i, j, k = slice(ib, ie), slice(jb, je), slice(kb, ke)
        ip1 = slice(ib + 1, ie + 1)

        if ow_tnd_s:
            tnd_s[i, j, k] = 1e-3 * s[i, j, k]
        else:
            tnd_s[i, j, k] += 1e-3 * s[i, j, k]

        if ow_tnd_su:
            tnd_su[i, j, k] = 300 * su[i, j, k]
        else:
            tnd_su[i, j, k] += 300 * su[i, j, k]

        if ow_tnd_u:
            tnd_u[i, j, k] = 50 * (u[i, j, k] + u[ip1, j, k])
        else:
            tnd_u[i, j, k] += 50 * (u[i, j, k] + u[ip1, j, k])

        fake[i, j, k] = 2 * s[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="fake1")
    def stencil_gt4py(
        s: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        tnd_s: gtscript.Field["dtype"],
        tnd_su: gtscript.Field["dtype"],
        tnd_u: gtscript.Field["dtype"],
        fake: gtscript.Field["dtype"],
        *,
        ow_tnd_s: bool,
        ow_tnd_su: bool,
        ow_tnd_u: bool
    ):
        with computation(PARALLEL), interval(...):
            if ow_tnd_s:
                tnd_s = 1e-3 * s
            else:
                tnd_s += 1e-3 * s

            if ow_tnd_su:
                tnd_su = 300 * su if ow_tnd_su else tnd_su + 300 * su
            else:
                tnd_su += 300 * su

            if ow_tnd_u:
                tnd_u = 50 * (u[0, 0, 0] + u[1, 0, 0])
            else:
                tnd_u += 50 * (u[0, 0, 0] + u[1, 0, 0])

            fake = 2 * s


@pytest.fixture(scope="module")
def make_fake_tendency_component_1():
    def _make_fake_tendency_component_1(
        domain,
        grid_type,
        *,
        backend=None,
        backend_options=None,
        storage_shape=None,
        storage_options=None
    ):
        return FakeTendencyComponent1(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

    return _make_fake_tendency_component_1


class FakeTendencyComponent2(TendencyComponent):
    def __init__(
        self,
        domain,
        grid_type,
        *,
        backend,
        backend_options,
        storage_shape,
        storage_options
    ):
        super().__init__(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.stencil = self.compile("fake2")

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "fake_variable": {"dims": dims, "units": "kg m^-2 K^-1"},
            "y_velocity_at_v_locations": {"dims": dims_y, "units": "m s^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(
        self, state, out_tendencies, out_diagnostics, overwrite_tendencies
    ):
        self.stencil(
            s=state["air_isentropic_density"],
            f=state["fake_variable"],
            v=state["y_velocity_at_v_locations"],
            tnd_s=out_tendencies["air_isentropic_density"],
            tnd_sv=out_tendencies["y_momentum_isentropic"],
            ow_tnd_s=overwrite_tendencies["air_isentropic_density"],
            ow_tnd_sv=overwrite_tendencies["y_momentum_isentropic"],
            origin=(0, 0, 0),
            domain=(self.grid.nx, self.grid.ny, self.grid.nz),
        )

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="fake2")
    def stencil_numpy(
        s, f, v, tnd_s, tnd_sv, *, ow_tnd_s, ow_tnd_sv, origin, domain
    ):
        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        i, j, k = slice(ib, ie), slice(jb, je), slice(kb, ke)
        jp1 = slice(jb + 1, je + 1)

        if ow_tnd_s:
            tnd_s[i, j, k] = 0.01 * f[i, j, k]
        else:
            tnd_s[i, j, k] += 0.01 * f[i, j, k]

        if ow_tnd_sv:
            tnd_sv[i, j, k] = 0.5 * s[i, j, k] * (v[i, j, k] + v[i, jp1, k])
        else:
            tnd_sv[i, j, k] += 0.5 * s[i, j, k] * (v[i, j, k] + v[i, jp1, k])

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="fake2")
    def stencil_gt4py(
        s: gtscript.Field["dtype"],
        f: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        tnd_s: gtscript.Field["dtype"],
        tnd_sv: gtscript.Field["dtype"],
        *,
        ow_tnd_s: bool,
        ow_tnd_sv: bool
    ):
        with computation(PARALLEL), interval(...):
            if ow_tnd_s:
                tnd_s = 0.01 * f[0, 0, 0]
            else:
                tnd_s += 0.01 * f[0, 0, 0]

            if ow_tnd_sv:
                tnd_sv = 0.5 * s[0, 0, 0] * (v[0, 0, 0] + v[0, 1, 0])
            else:
                tnd_sv += 0.5 * s[0, 0, 0] * (v[0, 0, 0] + v[0, 1, 0])


@pytest.fixture(scope="module")
def make_fake_tendency_component_2():
    def _make_fake_tendency_component_2(
        domain,
        grid_type,
        *,
        backend=None,
        backend_options=None,
        storage_shape=None,
        storage_options=None
    ):
        return FakeTendencyComponent2(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

    return _make_fake_tendency_component_2
