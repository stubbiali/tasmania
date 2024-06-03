# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from property_cached import cached_property
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.static_energy import (
    DryStaticEnergy,
    MoistStaticEnergy,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.suites.core_components import DiagnosticComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"


class DryStaticEnergyTestSuite(DiagnosticComponentTestSuite):
    def __init__(self, domain_suite, staggered):
        self.staggered = staggered
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return DryStaticEnergy(
            self.ds.domain,
            self.ds.grid_type,
            self.staggered,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        t = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz),
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="t",
        )
        h = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz),
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="h",
        )
        h_stg = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz + 1),
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="h_stg",
        )
        time = self.hyp_data.draw(hyp_st.datetimes(), label="time")
        state = {
            "time": time,
            "air_temperature": get_dataarray_3d(
                t,
                self.ds.grid,
                "K",
                grid_shape=(nx, ny, nz),
                set_coordinates=False,
            ),
            "height": get_dataarray_3d(
                h,
                self.ds.grid,
                "m",
                grid_shape=(nx, ny, nz),
                set_coordinates=False,
            ),
            "height_on_interface_levels": get_dataarray_3d(
                h_stg,
                self.ds.grid,
                "m",
                grid_shape=(nx, ny, nz + 1),
                set_coordinates=False,
            ),
        }
        return state

    def get_validation_diagnostics(self, raw_state_np):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        t = raw_state_np["air_temperature"][:nx, :ny, :nz]
        if self.staggered:
            aux = raw_state_np["height_on_interface_levels"]
            h = 0.5 * (aux[:nx, :ny, :nz] + aux[:nx, :ny, 1 : nz + 1])
        else:
            h = raw_state_np["height"][:nx, :ny, :nz]
        cp = self.component.rcp[
            "specific_heat_of_dry_air_at_constant_pressure"
        ]
        g = self.component.rcp["gravitational_acceleration"]
        dse = cp * t + g * h
        return {"montgomery_potential": dse}


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_dry(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = DryStaticEnergyTestSuite(ds, staggered=False)
    ts.run()
    ts = DryStaticEnergyTestSuite(ds, staggered=True)
    ts.run()


class MoistStaticEnergyTestSuite(DiagnosticComponentTestSuite):
    @cached_property
    def component(self):
        return MoistStaticEnergy(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        time = self.hyp_data.draw(hyp_st.datetimes(), label="time")
        mtg = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz),
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="mtg",
        )
        qv = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz),
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="qv",
        )
        state = {
            "time": time,
            "montgomery_potential": get_dataarray_3d(
                mtg,
                self.ds.grid,
                "m^2 s^-2",
                grid_shape=(nx, ny, nz),
                set_coordinates=False,
            ),
            mfwv: get_dataarray_3d(
                qv,
                self.ds.grid,
                "g g^-1",
                grid_shape=(nx, ny, nz),
                set_coordinates=False,
            ),
        }
        return state

    def get_validation_diagnostics(self, raw_state_np):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        mtg = raw_state_np["montgomery_potential"][:nx, :ny, :nz]
        qv = raw_state_np[mfwv][:nx, :ny, :nz]
        lhwv = self.component.rpc["latent_heat_of_vaporization_of_water"]
        mse = mtg + lhwv * qv
        return {"moist_static_energy": mse}


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_moist(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = MoistStaticEnergyTestSuite(ds)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
