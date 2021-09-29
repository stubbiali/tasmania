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
from hypothesis import (
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest
from sympl._core.units import units_are_same

from tasmania.python.framework.promoter import (
    FromTendencyToDiagnostic,
    FromDiagnosticToTendency,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import hyp_settings


class FakeTendency2Diagnostic(FromTendencyToDiagnostic):
    @property
    def input_properties(self):
        g = self._grid
        dim0, dim1, dim2 = g.x.dims[0], g.y.dims[0], g.z.dims[0]
        return_dict = {
            "air_pressure": {
                "dims": (dim0, dim1, dim2),  # (dim2, dim0, dim1),
                "units": "Pa s^-1",  # "hPa s^-1",
            },
            "x_velocity": {
                "dims": (dim0, dim1, dim2),
                "units": "m s^-2",
                "diagnostic_name": "tnd_of_x_velocity",
            },
            "y_velocity": {
                "dims": (dim0, dim1, dim2),  # (dim1, dim0, dim2),
                "units": "m s^-2",  # "km hr^-1 s^-1",
                "diagnostic_name": "y_velocity_abcde",
                "remove_from_tendencies": True,
            },
        }
        return return_dict


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
def test_tendency_to_diagnostic(data, backend):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 10),
            backend=backend,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    field = data.draw(
        st_raw_field((nx, ny, nz), -1e4, 1e4, backend=backend),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    dim0, dim1, dim2 = grid.x.dims[0], grid.y.dims[0], grid.z.dims[0]

    p = get_dataarray_3d(field, grid, "Pa s^-1", set_coordinates=False)
    u = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    v = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    w = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    tendencies = {
        "air_pressure": p,
        "x_velocity": u,
        "y_velocity": v,
        "z_velocity": w,
    }

    promoter = FakeTendency2Diagnostic(domain, grid_type)

    assert isinstance(promoter, Tendency2Diagnostic)

    assert "air_pressure" in promoter.input_properties
    ref = promoter.input_properties["air_pressure"]
    assert "tendency_of_air_pressure" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties["tendency_of_air_pressure"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert "x_velocity" in promoter.input_properties
    ref = promoter.input_properties["x_velocity"]
    assert "tnd_of_x_velocity" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties["tnd_of_x_velocity"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert "y_velocity" in promoter.input_properties
    ref = promoter.input_properties["y_velocity"]
    assert "y_velocity_abcde" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties["y_velocity_abcde"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert len(promoter.input_properties) == 3
    assert len(promoter.diagnostic_properties) == 3

    out = promoter(tendencies)

    assert "tendency_of_air_pressure" in out
    assert all(
        src == trg
        for src, trg in zip(
            out["tendency_of_air_pressure"].dims,
            (dim0, dim1, dim2),  # (dim2, dim0, dim1)
        )
    )
    assert units_are_same(
        out["tendency_of_air_pressure"].attrs["units"], "Pa s^-1"  # "hPa s^-1"
    )
    assert all(
        src == trg
        for src, trg in zip(
            out["tendency_of_air_pressure"].shape, (nx, ny, nz)  # (nz, nx, ny)
        )
    )

    assert "tnd_of_x_velocity" in out
    assert all(
        src == trg
        for src, trg in zip(out["tnd_of_x_velocity"].dims, (dim0, dim1, dim2))
    )
    assert units_are_same(out["tnd_of_x_velocity"].attrs["units"], "m s^-2")
    assert all(
        src == trg
        for src, trg in zip(out["tnd_of_x_velocity"].shape, (nx, ny, nz))
    )

    assert "y_velocity_abcde" in out
    assert all(
        src == trg
        for src, trg in zip(
            out["y_velocity_abcde"].dims,
            (dim0, dim1, dim2),  # (dim1, dim0, dim2)
        )
    )
    assert units_are_same(
        out["y_velocity_abcde"].attrs["units"], "m s^-2"  # "km s^-1 hr^-1"
    )
    assert all(
        src == trg
        for src, trg in zip(
            out["y_velocity_abcde"].shape, (nx, ny, nz)  # (ny, nx, nz)
        )
    )

    assert len(out) == 3

    assert "air_pressure" in tendencies
    assert "x_velocity" in tendencies
    assert "y_velocity" not in tendencies
    assert "z_velocity" in tendencies
    assert len(tendencies) == 3


class FakeDiagnostic2Tendency(FromDiagnosticToTendency):
    @property
    def input_properties(self):
        g = self._grid
        dim0, dim1, dim2 = g.x.dims[0], g.y.dims[0], g.z.dims[0]
        return_dict = {
            "air_pressure": {
                "dims": (dim0, dim1, dim2),  # (dim2, dim0, dim1),
                "units": "Pa s^-1",  # "hPa s^-1",
            },
            "tnd_of_x_velocity": {
                "dims": (dim0, dim1, dim2),
                "units": "m s^-2",
                "tnd_name": "x_velocity",
                "remove_from_diagnostics": False,
            },
            "tendency_of_y_velocity": {
                "dims": (dim0, dim1, dim2),  # (dim1, dim0, dim2),
                "units": "m s^-2",  # "km hr^-1 s^-1",
                "remove_from_diagnostics": True,
            },
        }
        return return_dict


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
def test_diagnostic_to_tendency(data, backend):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 10),
            backend=backend,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    field = data.draw(
        st_raw_field((nx, ny, nz), -1e4, 1e4, backend=backend),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    dim0, dim1, dim2 = grid.x.dims[0], grid.y.dims[0], grid.z.dims[0]

    p = get_dataarray_3d(field, grid, "Pa s^-1", set_coordinates=False)
    u = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    v = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    w = get_dataarray_3d(field, grid, "m s^-2", set_coordinates=False)
    diagnostics = {
        "air_pressure": p,
        "tnd_of_x_velocity": u,
        "tendency_of_y_velocity": v,
        "z_velocity": w,
    }

    promoter = FakeDiagnostic2Tendency(domain, grid_type)

    assert isinstance(promoter, FromDiagnosticToTendency)

    assert "air_pressure" in promoter.input_properties
    ref = promoter.input_properties["air_pressure"]
    assert "air_pressure" in promoter.tendency_properties
    check = promoter.tendency_properties["air_pressure"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert "tnd_of_x_velocity" in promoter.input_properties
    ref = promoter.input_properties["tnd_of_x_velocity"]
    assert "tnd_of_x_velocity" in promoter.tendency_properties
    check = promoter.tendency_properties["tnd_of_x_velocity"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert "tendency_of_y_velocity" in promoter.input_properties
    ref = promoter.input_properties["tendency_of_y_velocity"]
    assert "y_velocity" in promoter.tendency_properties
    check = promoter.tendency_properties["y_velocity"]
    assert check["dims"] == ref["dims"]
    assert check["units"] == ref["units"]

    assert len(promoter.input_properties) == 3
    assert len(promoter.tendency_properties) == 3

    out = promoter(diagnostics)

    assert "air_pressure" in out
    assert all(
        src == trg
        for src, trg in zip(
            out["air_pressure"].dims, (dim0, dim1, dim2)  # (dim2, dim0, dim1)
        )
    )
    assert units_are_same(
        out["air_pressure"].attrs["units"], "Pa s^-1"
    )  # "hPa s^-1"
    assert all(
        src == trg
        for src, trg in zip(
            out["air_pressure"].shape, (nx, ny, nz)  # (nz, nx, ny)
        )
    )

    assert "tnd_of_x_velocity" in out
    assert all(
        src == trg
        for src, trg in zip(out["tnd_of_x_velocity"].dims, (dim0, dim1, dim2))
    )
    assert units_are_same(out["tnd_of_x_velocity"].attrs["units"], "m s^-2")
    assert all(
        src == trg
        for src, trg in zip(out["tnd_of_x_velocity"].shape, (nx, ny, nz))
    )

    assert "y_velocity" in out
    assert all(
        src == trg
        for src, trg in zip(
            out["y_velocity"].dims, (dim0, dim1, dim2)  # (dim1, dim0, dim2)
        )
    )
    assert units_are_same(
        out["y_velocity"].attrs["units"], "m s^-2"  # "km s^-1 hr^-1"
    )
    assert all(
        src == trg
        for src, trg in zip(
            out["y_velocity"].shape, (nx, ny, nz)  # (ny, nx, nz)
        )
    )

    assert len(out) == 3

    assert "air_pressure" in diagnostics
    assert "tnd_of_x_velocity" in diagnostics
    assert "tendency_of_y_velocity" not in diagnostics
    assert "z_velocity" in diagnostics
    assert len(diagnostics) == 3


if __name__ == "__main__":
    pytest.main([__file__])
