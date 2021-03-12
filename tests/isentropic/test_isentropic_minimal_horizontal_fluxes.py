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

from tests import conf
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
@pytest.mark.parametrize("flux_scheme", flux_properties.keys())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, flux_scheme, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="field",
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        IsentropicMinimalHorizontalFlux,
        flux_scheme,
        domain,
        field,
        timestep,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_centered(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
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
            aligned_index=aligned_index,
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
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)
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
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_third_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf.nb)), label="nb"
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
            aligned_index=aligned_index,
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
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)
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
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_fifth_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
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
            aligned_index=aligned_index,
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
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)
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
