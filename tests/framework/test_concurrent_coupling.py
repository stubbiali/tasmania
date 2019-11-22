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
    assume,
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import numpy as np
import pytest
from sympl._core.exceptions import InvalidStateError

import gt4py as gt

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict

try:
    from .conf import backend as conf_backend, default_origin as conf_dorigin
    from .utils import (
        compare_arrays,
        st_domain,
        st_isentropic_state_f,
        st_one_of,
        st_timedeltas,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, default_origin as conf_dorigin
    from utils import (
        compare_arrays,
        st_domain,
        st_isentropic_state_f,
        st_one_of,
        st_timedeltas,
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
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
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60))
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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def test_serial(data, make_fake_tendency_component_1, make_fake_tendency_component_2):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20)),
        label="domain",
    )
    cgrid = domain.numerical_grid

    state = data.draw(
        st_isentropic_state_f(cgrid, moist=False, precipitation=False), label="state"
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60))
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")
    tc2 = make_fake_tendency_component_2(domain, "numerical")

    cc = ConcurrentCoupling(tc1, tc2, execution_policy="serial", gt_powered=False)
    tendencies, diagnostics = cc(state, dt)

    assert "fake_variable" in diagnostics
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = diagnostics["fake_variable"].to_units("kg m^-2 K^-1").values
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in tendencies
    compare_arrays(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values,
        1e-3 * s + 1e-2 * f,
    )

    assert "x_momentum_isentropic" in tendencies
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    compare_arrays(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values, 300 * su
    )

    assert "y_momentum_isentropic" in tendencies
    v = state["y_velocity_at_v_locations"].to_units("m s^-1").values
    compare_arrays(
        tendencies["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values,
        0.5 * s * (v[:, :-1, :] + v[:, 1:, :]),
    )

    assert "x_velocity_at_u_locations" in tendencies
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values
    compare_arrays(
        tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values, 50 * u
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def test_gt_serial(data, make_fake_tendency_component_1, make_fake_tendency_component_2):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20)),
        label="domain",
    )
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = cgrid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    storage_shape = (cgrid.nx + 1, cgrid.ny + 1, cgrid.nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60))
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")
    tc2 = make_fake_tendency_component_2(domain, "numerical")

    cc = ConcurrentCoupling(
        tc1,
        tc2,
        execution_policy="serial",
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        rebuild=False,
    )
    tendencies, diagnostics = cc(state, dt)

    assert "fake_variable" in diagnostics
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = diagnostics["fake_variable"].to_units("kg m^-2 K^-1").values
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in tendencies
    compare_arrays(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values,
        1e-3 * s + 1e-2 * f,
    )

    assert "x_momentum_isentropic" in tendencies
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    compare_arrays(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values, 300 * su
    )

    assert "y_momentum_isentropic" in tendencies
    v = state["y_velocity_at_v_locations"].to_units("m s^-1").values
    v_val = np.zeros(storage_shape, dtype=cgrid.x.dtype)
    v_val[:-1, :-1, :-1] = 0.5 * s[:-1, :-1, :-1] * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
    compare_arrays(
        tendencies["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-2")
        .values[:-1, :-1, :-1],
        v_val[:-1, :-1, :-1],
    )

    assert "x_velocity_at_u_locations" in tendencies
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values
    compare_arrays(
        tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values, 50 * u
    )


if __name__ == "__main__":
    pytest.main([__file__])
