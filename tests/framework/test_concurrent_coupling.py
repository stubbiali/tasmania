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
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

from sympl._core.exceptions import InvalidArrayDictError

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.promoter import FromTendencyToDiagnostic

from tests import conf
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)
from tests.suites.concurrent_coupling import ConcurrentCouplingTestSuite
from tests.suites.core_components import (
    FakeTendencyComponent1TestSuite,
    FakeTendencyComponent2TestSuite,
)
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_compatibility(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    ds = DomainSuite(data, backend="numpy", dtype=float)
    tcts1 = FakeTendencyComponent1TestSuite(ds)
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    state = tcts1.get_state()

    # failing
    ccts = ConcurrentCouplingTestSuite(
        ds, tcts1, tcts2, execution_policy="as_parallel"
    )
    with pytest.raises(InvalidArrayDictError):
        ccts.run(state)

    # failing
    ccts = ConcurrentCouplingTestSuite(
        ds, tcts2, tcts1, execution_policy="serial"
    )
    with pytest.raises(InvalidArrayDictError):
        ccts.run(state)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_serial(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    ds = DomainSuite(data, backend, dtype, grid_type="numerical")
    tcts1 = FakeTendencyComponent2TestSuite(ds)
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    ccts = ConcurrentCouplingTestSuite(
        ds, tcts1, tcts2, execution_policy="serial"
    )
    ccts.run()


class FakeTendency2Diagnostic(FromTendencyToDiagnostic):
    def __init__(self, domain):
        super().__init__(domain, "numerical")

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "x_velocity": {
                "dims": dims,
                "units": "m s^-2",
                "diagnostic_name": "tnd_of_x_velocity",
            },
        }

        return return_dict


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def _test_tendency_to_diagnostic(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    ngrid = domain.numerical_grid
    nx, ny, nz = ngrid.nx, ngrid.ny, ngrid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            ngrid,
            moist=False,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
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
    tc1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    tc2 = make_fake_tendency_component_2(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
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
    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    f = to_numpy(diagnostics["fake_variable"].to_units("kg m^-2 K^-1").data)
    i, j, k = slice(0, nx), slice(0, ny), slice(0, nz)
    compare_arrays(f, 2 * s, slice=(i, j, k))

    assert "air_isentropic_density" in tendencies
    compare_arrays(
        tendencies["air_isentropic_density"]
        .to_units("kg m^-2 K^-1 s^-1")
        .data,
        1e-3 * s + 1e-2 * f,
        slice=(i, j, k),
    )

    assert "tendency_of_air_isentropic_density" in diagnostics
    compare_arrays(
        diagnostics["tendency_of_air_isentropic_density"]
        .to_units("kg m^-2 K^-1 s^-1")
        .data,
        1e-3 * s + 1e-2 * f,
        slice=(i, j, k),
    )

    assert "x_momentum_isentropic" in tendencies
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    compare_arrays(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data,
        300 * su,
        slice=(i, j, k),
    )

    assert "y_momentum_isentropic" in tendencies
    v = to_numpy(state["y_velocity_at_v_locations"].to_units("m s^-1").data)
    v_val = 0.5 * s[:, :-1] * (v[:, :-1] + v[:, 1:])
    compare_arrays(
        tendencies["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data,
        v_val,
        slice=(i, j, k),
    )

    assert "x_velocity" in tendencies
    u = to_numpy(state["x_velocity_at_u_locations"].to_units("m s^-1").data)
    compare_arrays(
        tendencies["x_velocity"].to_units("m s^-2").data,
        50 * (u[:-1] + u[1:]),
        slice=(i, j, k),
    )

    assert "tnd_of_x_velocity" in diagnostics
    compare_arrays(
        diagnostics["tnd_of_x_velocity"].to_units("m s^-2").data,
        50 * (u[:-1] + u[1:]),
        slice=(i, j, k),
    )

    assert len(tendencies) == 5
    assert len(diagnostics) == 4


if __name__ == "__main__":
    pytest.main([__file__])
