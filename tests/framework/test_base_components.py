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
from hypothesis import (
    assume,
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py as gt

from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    TendencyComponent,
    TendencyPromoter,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d

try:
    from .utils import compare_arrays, st_domain, st_one_of, st_raw_field
except (ImportError, ModuleNotFoundError):
    from utils import compare_arrays, st_domain, st_one_of, st_raw_field


class FakeDiagnosticComponent(DiagnosticComponent):
    def __init__(self, domain, grid_type, **kwargs):
        super().__init__(domain, grid_type)

    @property
    def input_properties(self):
        return {}

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        return {}


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_diagnostic_component(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    # ========================================
    # test bed
    # ========================================
    obj = FakeDiagnosticComponent(domain, "physical")
    assert isinstance(obj, DiagnosticComponent)

    obj = FakeDiagnosticComponent(domain, "numerical")
    assert isinstance(obj, DiagnosticComponent)


class FakeImplicitTendencyComponent(ImplicitTendencyComponent):
    def __init__(self, domain, grid_type, **kwargs):
        super().__init__(domain, grid_type, **kwargs)

    @property
    def input_properties(self):
        return {}

    @property
    def tendency_properties(self):
        return {}

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state, timestep):
        return {}, {}


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_implicit_tendency_component(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    # ========================================
    # test bed
    # ========================================
    obj = FakeImplicitTendencyComponent(domain, "physical")
    assert isinstance(obj, ImplicitTendencyComponent)

    obj = FakeImplicitTendencyComponent(domain, "numerical")
    assert isinstance(obj, ImplicitTendencyComponent)


class FakeTendencyComponent(TendencyComponent):
    def __init__(self, domain, grid_type, **kwargs):
        super().__init__(domain, grid_type, **kwargs)

    @property
    def input_properties(self):
        return {}

    @property
    def tendency_properties(self):
        return {}

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        return {}, {}


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_tendency_component(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    # ========================================
    # test bed
    # ========================================
    obj = FakeTendencyComponent(domain, "physical")
    assert isinstance(obj, TendencyComponent)

    obj = FakeTendencyComponent(domain, "numerical")
    assert isinstance(obj, TendencyComponent)


class FakeTendencyPromoter(TendencyPromoter):
    @property
    def input_properties(self):
        g = self._grid
        dim0, dim1, dim2 = g.x.dims[0], g.y.dims[0], g.z.dims[0]
        return_dict = {
            "air_pressure": {"dims": (dim2, dim0, dim1), "units": "hPa"},
            "x_velocity": {
                "dims": (dim0, dim1, dim2),
                "units": "m s^-1",
                "prefix": "tnd_of_",
            },
            "y_velocity": {
                "dims": (dim1, dim0, dim2),
                "units": "km hr^-1",
                "suffix": "_abcde",
            },
        }
        return return_dict


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_tendency_promoter(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 10)),
        label="domain",
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    field = data.draw(
        st_raw_field(
            (grid.nx, grid.ny, grid.nz), -1e4, 1e4, "numpy", grid.x.dtype, (0, 0, 0)
        ),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    dim0, dim1, dim2 = grid.x.dims[0], grid.y.dims[0], grid.z.dims[0]

    p = get_dataarray_3d(field, grid, "Pa", set_coordinates=False)
    u = get_dataarray_3d(field, grid, "m s^-1", set_coordinates=False)
    v = get_dataarray_3d(field, grid, "m s^-1", set_coordinates=False)
    w = get_dataarray_3d(field, grid, "m s^-1", set_coordinates=False)
    tendencies = {"air_pressure": p, "x_velocity": u, "y_velocity": v, "z_velocity": w}

    promoter = FakeTendencyPromoter(domain, grid_type)

    assert isinstance(promoter, TendencyPromoter)

    assert "air_pressure" in promoter.input_properties
    ref = promoter.input_properties["air_pressure"]
    assert "tendency_of_air_pressure" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties['tendency_of_air_pressure']
    assert check['dims'] == ref['dims']
    assert check['units'] == ref['units']

    assert "x_velocity" in promoter.input_properties
    ref = promoter.input_properties["x_velocity"]
    assert "tnd_of_x_velocity" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties['tnd_of_x_velocity']
    assert check['dims'] == ref['dims']
    assert check['units'] == ref['units']

    assert "y_velocity" in promoter.input_properties
    ref = promoter.input_properties["y_velocity"]
    assert "tendency_of_y_velocity_abcde" in promoter.diagnostic_properties
    check = promoter.diagnostic_properties['tendency_of_y_velocity_abcde']
    assert check['dims'] == ref['dims']
    assert check['units'] == ref['units']

    assert len(promoter.input_properties) == 3
    assert len(promoter.diagnostic_properties) == 3

    out = promoter(tendencies)

    assert "tendency_of_air_pressure" in out
    assert all(
        src == trg
        for src, trg in zip(out["tendency_of_air_pressure"].dims, (dim2, dim0, dim1))
    )
    assert out["tendency_of_air_pressure"].attrs["units"] == "hPa"
    assert all(
        src == trg
        for src, trg in zip(
            out["tendency_of_air_pressure"].shape, (grid.nz, grid.nx, grid.ny)
        )
    )

    assert "tnd_of_x_velocity" in out
    assert all(
        src == trg for src, trg in zip(out["tnd_of_x_velocity"].dims, (dim0, dim1, dim2))
    )
    assert out["tnd_of_x_velocity"].attrs["units"] == "m s^-1"
    assert all(
        src == trg
        for src, trg in zip(out["tnd_of_x_velocity"].shape, (grid.nx, grid.ny, grid.nz))
    )

    assert "tendency_of_y_velocity_abcde" in out
    assert all(
        src == trg
        for src, trg in zip(out["tendency_of_y_velocity_abcde"].dims, (dim1, dim0, dim2))
    )
    assert out["tendency_of_y_velocity_abcde"].attrs["units"] == "km hr^-1"
    assert all(
        src == trg
        for src, trg in zip(
            out["tendency_of_y_velocity_abcde"].shape, (grid.ny, grid.nx, grid.nz)
        )
    )

    assert len(out) == 3


if __name__ == "__main__":
    pytest.main([__file__])
