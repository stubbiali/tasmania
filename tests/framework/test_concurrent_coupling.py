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
from copy import deepcopy
from datetime import timedelta
from hypothesis import (
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest
from sympl._core.exceptions import InvalidStateError

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.promoters import Tendency2Diagnostic
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict
from tasmania.python.utils.utils import is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_compatibility(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=True,
            precipitation=True,
            backend="numpy",
            storage_shape=(nx + 1, ny + 1, nz + 1),
        ),
        label="state",
    )

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        )
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")
    tc2 = make_fake_tendency_component_2(domain, "numerical")

    #
    # failing
    #
    state_dc = deepcopy_dataarray_dict(state)
    cc1 = ConcurrentCoupling(tc1, tc2, execution_policy="as_parallel")
    try:
        cc1(state_dc, dt)
        assert False
    except InvalidStateError:
        assert True

    #
    # failing
    #
    state_dc = deepcopy_dataarray_dict(state)
    cc2 = ConcurrentCoupling(tc2, tc1, execution_policy="serial")
    try:
        cc2(state_dc, dt)
        assert False
    except InvalidStateError:
        assert True

    #
    # successful
    #
    state_dc = deepcopy_dataarray_dict(state)
    cc3 = ConcurrentCoupling(tc1, tc2, execution_policy="serial")
    try:
        cc3(state_dc, dt)
        assert True
    except InvalidStateError:
        assert False


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_serial(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    cgrid = domain.numerical_grid

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (cgrid.nx + dnx, cgrid.ny + dny, cgrid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        )
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    tc1 = make_fake_tendency_component_1(domain, "numerical")
    tc2 = make_fake_tendency_component_2(domain, "numerical")

    cc = ConcurrentCoupling(
        tc1,
        tc2,
        execution_policy="serial",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    tendencies, diagnostics = cc(state, dt)

    assert "fake_variable" in diagnostics
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    f = diagnostics["fake_variable"].to_units("kg m^-2 K^-1").data
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in tendencies
    compare_arrays(
        tendencies["air_isentropic_density"]
        .to_units("kg m^-2 K^-1 s^-1")
        .data,
        1e-3 * s + 1e-2 * f,
    )

    assert "x_momentum_isentropic" in tendencies
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    compare_arrays(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data,
        300 * su,
    )

    assert "y_momentum_isentropic" in tendencies
    v = state["y_velocity_at_v_locations"].to_units("m s^-1").data
    if same_shape or is_gt(backend):
        v_val = deepcopy(v)
        v_val[:-1, :-1, :-1] = (
            0.5 * s[:-1, :-1, :-1] * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
        )
        compare_arrays(
            tendencies["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-2")
            .data[:-1, :-1, :-1],
            v_val[:-1, :-1, :-1],
        )
    else:
        v_val = 0.5 * s * (v[:, :-1] + v[:, 1:])
        compare_arrays(
            tendencies["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-2")
            .data,
            v_val,
        )

    assert "x_velocity_at_u_locations" in tendencies
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").data
    compare_arrays(
        tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data,
        50 * u,
    )


class FakeTendency2Diagnostic(Tendency2Diagnostic):
    def __init__(self, domain):
        super().__init__(domain, "numerical")

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgx = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "x_velocity_at_u_locations": {
                "dims": dims_stgx,
                "units": "m s^-2",
                "diagnostic_name": "tnd_of_x_velocity_at_u_locations",
            },
        }

        return return_dict


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_tendency_to_diagnostic(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    cgrid = domain.numerical_grid

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (cgrid.nx + dnx, cgrid.ny + dny, cgrid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        )
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    tc1 = make_fake_tendency_component_1(domain, "numerical")
    tc2 = make_fake_tendency_component_2(domain, "numerical")
    t2d = FakeTendency2Diagnostic(domain)

    cc = ConcurrentCoupling(
        tc1,
        tc2,
        t2d,
        execution_policy="serial",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    tendencies, diagnostics = cc(state, dt)

    assert "fake_variable" in diagnostics
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    f = diagnostics["fake_variable"].to_units("kg m^-2 K^-1").data
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in tendencies
    compare_arrays(
        tendencies["air_isentropic_density"]
        .to_units("kg m^-2 K^-1 s^-1")
        .data,
        1e-3 * s + 1e-2 * f,
    )

    assert "tendency_of_air_isentropic_density" in diagnostics
    compare_arrays(
        diagnostics["tendency_of_air_isentropic_density"]
        .to_units("kg m^-2 K^-1 s^-1")
        .data,
        1e-3 * s + 1e-2 * f,
    )

    assert "x_momentum_isentropic" in tendencies
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    compare_arrays(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data,
        300 * su,
    )

    assert "y_momentum_isentropic" in tendencies
    v = state["y_velocity_at_v_locations"].to_units("m s^-1").data
    if same_shape or is_gt(backend):
        v_val = deepcopy(v)
        v_val[:-1, :-1, :-1] = (
            0.5 * s[:-1, :-1, :-1] * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
        )
        compare_arrays(
            tendencies["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-2")
            .data[:-1, :-1, :-1],
            v_val[:-1, :-1, :-1],
        )
    else:
        v_val = 0.5 * s * (v[:, :-1] + v[:, 1:])
        compare_arrays(
            tendencies["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-2")
            .data,
            v_val,
        )

    assert "x_velocity_at_u_locations" in tendencies
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").data
    compare_arrays(
        tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data,
        50 * u,
    )

    assert "tnd_of_x_velocity_at_u_locations" in diagnostics
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").data
    compare_arrays(
        diagnostics["tnd_of_x_velocity_at_u_locations"]
        .to_units("m s^-2")
        .data,
        50 * u,
    )

    assert len(tendencies) == 5
    assert len(diagnostics) == 4


if __name__ == "__main__":
    pytest.main([__file__])
