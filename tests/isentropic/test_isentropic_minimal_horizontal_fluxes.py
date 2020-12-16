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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.subclasses.minimal_horizontal_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.isentropic.test_isentropic_horizontal_fluxes import (
    get_centered_fluxes,
    get_fifth_order_upwind_fluxes,
    get_third_order_upwind_fluxes,
    get_upwind_fluxes,
    validation,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import hyp_settings


def test_registry():
    assert "upwind" in IsentropicMinimalHorizontalFlux.registry
    assert IsentropicMinimalHorizontalFlux.registry["upwind"] == Upwind
    assert "centered" in IsentropicMinimalHorizontalFlux.registry
    assert IsentropicMinimalHorizontalFlux.registry["centered"] == Centered
    assert "third_order_upwind" in IsentropicMinimalHorizontalFlux.registry
    assert (
        IsentropicMinimalHorizontalFlux.registry["third_order_upwind"]
        == ThirdOrderUpwind
    )
    assert "fifth_order_upwind" in IsentropicMinimalHorizontalFlux.registry
    assert (
        IsentropicMinimalHorizontalFlux.registry["fifth_order_upwind"]
        == FifthOrderUpwind
    )


def test_factory():
    obj = IsentropicMinimalHorizontalFlux.factory("upwind", backend="numpy")
    assert isinstance(obj, Upwind)
    obj = IsentropicMinimalHorizontalFlux.factory("centered", backend="numpy")
    assert isinstance(obj, Centered)
    obj = IsentropicMinimalHorizontalFlux.factory(
        "third_order_upwind", backend="numpy"
    )
    assert isinstance(obj, ThirdOrderUpwind)
    obj = IsentropicMinimalHorizontalFlux.factory(
        "fifth_order_upwind", backend="numpy"
    )
    assert isinstance(obj, FifthOrderUpwind)


flux_properties = {
    "upwind": {"type": Upwind, "get_fluxes": get_upwind_fluxes},
    "centered": {"type": Centered, "get_fluxes": get_centered_fluxes},
    "third_order_upwind": {
        "type": ThirdOrderUpwind,
        "get_fluxes": get_third_order_upwind_fluxes,
    },
    "fifth_order_upwind": {
        "type": FifthOrderUpwind,
        "get_fluxes": get_fifth_order_upwind_fluxes,
    },
}


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicMinimalHorizontalFlux,
        "upwind",
        domain,
        field,
        timestep,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_centered(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicMinimalHorizontalFlux,
        "centered",
        domain,
        field,
        timestep,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_third_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicMinimalHorizontalFlux,
        "third_order_upwind",
        domain,
        field,
        timestep,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_fifth_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicMinimalHorizontalFlux,
        "fifth_order_upwind",
        domain,
        field,
        timestep,
        backend,
        bo,
        so,
    )


if __name__ == "__main__":
    pytest.main([__file__])
