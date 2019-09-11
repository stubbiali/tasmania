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
from datetime import datetime
from hypothesis import (
    assume,
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gridtools as gt
from tasmania.python.framework.ng_base_components import NGTendencyComponent
from tasmania.python.utils.storage_utils import (
    get_storage_descriptor,
    make_dataarray_3d,
)

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import compare_dataarrays, st_arrays, st_domain, st_floats, st_one_of
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import compare_dataarrays, st_arrays, st_domain, st_floats, st_one_of


class FakeTendencyComponent(NGTendencyComponent):
    def __init__(self, domain, grid_type, *, backend, dtype, halo, rebuild, **kwargs):
        super().__init__(domain, grid_type, **kwargs)

        storage_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo)
        self.out_c = gt.storage.empty(descriptor, backend=backend)
        self.out_d = gt.storage.empty(descriptor, backend=backend)

        decorator = gt.stencil(backend, rebuild=rebuild)
        self.stencil = decorator(self.stencil_defs)

    @property
    def input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return_dict = {
            "field_a": {"dims": dims, "units": "kg"},
            "field_b": {"dims": dims, "units": "kg"},
        }
        return return_dict

    @property
    def tendency_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return_dict = {"field_c": {"dims": dims, "units": "kg s^-1"}}
        return return_dict

    @property
    def diagnostic_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return_dict = {"field_d": {"dims": dims, "units": "kg"}}
        return return_dict

    def array_call(self, state):
        in_a = state["field_a"]
        in_b = state["field_b"]

        self.stencil(
            in_a=in_a,
            in_b=in_b,
            out_c=self.out_c,
            out_d=self.out_d,
            origin={"_all_": (0, 0, 0)},
            domain=in_a.data.shape,
        )

        tendencies = {"field_c": self.out_c}
        diagnostics = {"field_d": self.out_d}

        return tendencies, diagnostics

    @staticmethod
    def stencil_defs(
        in_a: gt.storage.f64_sd,
        in_b: gt.storage.f64_sd,
        out_c: gt.storage.f64_sd,
        out_d: gt.storage.f64_sd,
    ):
        out_c = in_a[0, 0, 0] + in_b[0, 0, 0]
        out_d = in_a[0, 0, 0] - in_b[0, 0, 0]


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_tendency_component(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)))

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30), nb=nb
        ),
        label="domain",
    )

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    dtype = grid.x.dtype
    phi = data.draw(
        st_arrays(
            grid.x.dtype,
            (grid.nx, grid.ny, grid.nz + 1),
            elements=st_floats(min_value=-1e10, max_value=1e10),
            fill=hyp_st.nothing(),
        ),
        label="phi",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test
    # ========================================
    storage_shape = (grid.nx, grid.ny, grid.nz)
    descriptor = get_storage_descriptor(storage_shape, dtype, halo)

    field_a_st = gt.storage.from_array(phi[:, :, :-1], descriptor, backend=backend)
    field_a = make_dataarray_3d(field_a_st.data, grid, "kg")
    field_a.attrs["gt_storage"] = field_a_st
    field_b_st = gt.storage.from_array(phi[:, :, 1:], descriptor, backend=backend)
    field_b = make_dataarray_3d(field_b_st.data, grid, "kg")
    field_b.attrs["gt_storage"] = field_b_st
    state = {
        "time": datetime(year=1992, month=2, day=20),
        "field_a": field_a,
        "field_b": field_b,
    }

    ftc = FakeTendencyComponent(
        domain, grid_type, backend=backend, dtype=dtype, halo=halo, rebuild=False
    )

    tendencies, diagnostics = ftc(state)

    out_c_val = make_dataarray_3d(field_a.values + field_b.values, grid, "kg s^-1")
    assert "field_c" in tendencies
    compare_dataarrays(
        tendencies["field_c"], out_c_val, compare_coordinate_values=False
    )
    assert id(tendencies["field_c"].attrs["gt_storage"]) == id(ftc.out_c)
    assert id(tendencies["field_c"].values) == id(ftc.out_c.data)
    assert len(tendencies) == 1

    out_d_val = make_dataarray_3d(field_a.values - field_b.values, grid, "kg")
    assert "field_d" in diagnostics
    compare_dataarrays(
        diagnostics["field_d"], out_d_val, compare_coordinate_values=False
    )
    assert id(diagnostics["field_d"].attrs["gt_storage"]) == id(ftc.out_d)
    assert id(diagnostics["field_d"].values) == id(ftc.out_d.data)
    assert len(diagnostics) == 1


if __name__ == "__main__":
    pytest.main([__file__])
