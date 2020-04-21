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
from datetime import timedelta
from hypothesis import (
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py as gt

from tasmania.python.framework.tendency_stepper import TendencyStepper
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)
from tests.utilities import compare_dataarrays


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_implicit(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    if gt_powered:
        gt.storage.prepare_numpy()

    domain = data.draw(
        st_domain(
            zaxis_length=(2, 30), gt_powered=gt_powered, backend=backend, dtype=dtype
        ),
        label="domain",
    )
    cgrid = domain.numerical_grid

    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=True,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, cgrid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    iva = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist=True,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )

    imp = TendencyStepper.factory(
        "implicit",
        iva,
        execution_policy="serial",
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
    )

    assert imp.input_properties == iva.input_properties
    assert imp.output_properties == imp.output_properties

    diagnostics, out_state = imp(state, dt)

    iva_val = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist=True,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    _, diagnostics_val = iva_val(state, dt)

    diagnostics.pop("time", None)
    assert len(diagnostics) == 0

    out_state.pop("time", None)
    assert len(out_state) == len(diagnostics_val)
    for name in out_state:
        # with subtests.test(name=name):
        assert name in diagnostics_val
        compare_dataarrays(
            out_state[name], diagnostics_val[name], compare_coordinate_values=False
        )


if __name__ == "__main__":
    pytest.main([__file__])
