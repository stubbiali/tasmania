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
import abc
import numpy as np
from property_cached import cached_property
from typing import Optional, Dict, TYPE_CHECKING, Tuple

from sympl._core.units import units_are_same

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.utils.storage import (
    deepcopy_dataarray_dict,
    get_dataarray_3d,
)

from tests import conftest
from tests.strategies import (
    st_out_diagnostics,
    st_out_tendencies,
    st_overwrite_tendencies,
    st_isentropic_state_f,
    st_raw_field,
)
from tests.suites.component import ComponentTestSuite
from tests.suites.domain import DomainSuite

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLikeDict

    from tasmania.python.utils.typingx import (
        DiagnosticComponent,
        TendencyComponent,
        TimeDelta,
    )


class DiagnosticComponentTestSuite(ComponentTestSuite):
    def __init__(self, domain_suite: DomainSuite) -> None:
        super().__init__(domain_suite)

        self.input_properties = self.component.input_properties
        self.diagnostic_properties = self.component.diagnostic_properties

        self.out = self.get_out()
        self.out_dc = (
            deepcopy_dataarray_dict(self.out) if self.out is not None else {}
        )

    @cached_property
    @abc.abstractmethod
    def component(self) -> "DiagnosticComponent":
        pass

    def get_out(self) -> Optional["DataArrayDict"]:
        return self.hyp_data.draw(st_out_diagnostics(self.component))

    def run(self, state: Optional["DataArrayDict"] = None):
        state = state or self.get_state()

        diagnostics = self.component(state, out=self.out)

        raw_state_np = {
            name: to_numpy(
                state[name].to_units(self.input_properties[name]["units"]).data
            )
            for name in self.input_properties
        }
        raw_state_np["time"] = state["time"]
        raw_diagnostics_np = self.get_validation_diagnostics(raw_state_np)

        for name in self.diagnostic_properties:
            assert name in diagnostics
            assert units_are_same(
                diagnostics[name].attrs["units"],
                self.diagnostic_properties[name]["units"],
            )
            self.assert_allclose(
                name, diagnostics[name].data, raw_diagnostics_np[name]
            )

    @abc.abstractmethod
    def get_validation_diagnostics(
        self, raw_state_np: "NDArrayLikeDict"
    ) -> "NDArrayLikeDict":
        pass


class TendencyComponentTestSuite(ComponentTestSuite):
    def __init__(self, domain_suite: DomainSuite) -> None:
        super().__init__(domain_suite)

        self.input_properties = self.component.input_properties
        self.tendency_properties = self.component.tendency_properties
        self.diagnostic_properties = self.component.diagnostic_properties

        self.out_tendencies = self.get_out_tendencies()
        self.out_diagnostics = self.get_out_diagnostics()
        self.overwrite_tendencies = self.get_overwrite_tendencies()

        self.out_tendencies_dc = (
            deepcopy_dataarray_dict(self.out_tendencies)
            if self.out_tendencies is not None
            else {}
        )
        self.out_diagnostics_dc = (
            deepcopy_dataarray_dict(self.out_diagnostics)
            if self.out_diagnostics is not None
            else {}
        )
        overwrite_tendencies = self.overwrite_tendencies or {}
        self.overwrite_tendencies_dc = {
            name: overwrite_tendencies.get(name, True)
            or name not in self.out_tendencies_dc
            for name in self.tendency_properties
        }

    @cached_property
    @abc.abstractmethod
    def component(self) -> "TendencyComponent":
        pass

    def get_out_tendencies(self) -> Optional["DataArrayDict"]:
        return self.hyp_data.draw(st_out_tendencies(self.component))

    def get_out_diagnostics(self) -> Optional["DataArrayDict"]:
        return self.hyp_data.draw(st_out_diagnostics(self.component))

    def get_overwrite_tendencies(self) -> Optional[Dict[str, bool]]:
        return self.hyp_data.draw(st_overwrite_tendencies(self.component))

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ):
        state = state or self.get_state()

        try:
            tendencies, diagnostics = self.component(
                state,
                out_tendencies=self.out_tendencies,
                out_diagnostics=self.out_diagnostics,
                overwrite_tendencies=self.overwrite_tendencies,
            )
        except TypeError:
            timestep = timestep or self.get_timestep()
            tendencies, diagnostics = self.component(
                state,
                timestep,
                out_tendencies=self.out_tendencies,
                out_diagnostics=self.out_diagnostics,
                overwrite_tendencies=self.overwrite_tendencies,
            )

        raw_state_np = {
            name: to_numpy(
                state[name].to_units(self.input_properties[name]["units"]).data
            )
            for name in self.input_properties
        }
        raw_state_np["time"] = state["time"]
        (
            raw_tendencies_np,
            raw_diagnostics_np,
        ) = self.get_validation_tendencies_and_diagnostics(
            raw_state_np,
            dt=timestep.total_seconds() if timestep is not None else None,
        )

        for name in self.tendency_properties:
            assert name in tendencies
            assert units_are_same(
                tendencies[name].attrs["units"],
                self.tendency_properties[name]["units"],
            )
            if self.overwrite_tendencies_dc[name]:
                val = raw_tendencies_np[name]
            else:
                val = (
                    to_numpy(self.out_tendencies_dc[name].data)
                    + raw_tendencies_np[name]
                )
            self.assert_allclose(name, tendencies[name].data, val)

        for name in self.diagnostic_properties:
            assert name in diagnostics
            assert units_are_same(
                diagnostics[name].attrs["units"],
                self.diagnostic_properties[name]["units"],
            )
            self.assert_allclose(
                name, diagnostics[name].data, raw_diagnostics_np[name]
            )

    @abc.abstractmethod
    def get_validation_tendencies_and_diagnostics(
        self, raw_state_np: "NDArrayLikeDict", dt: Optional[float]
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        pass


class FakeTendencyComponent1TestSuite(TendencyComponentTestSuite):
    @cached_property
    def component(self) -> "TendencyComponent":
        return conftest.FakeTendencyComponent1(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.storage_options,
        )

    def get_state(self) -> "DataArrayDict":
        return self.ds.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )
        )

    def get_validation_tendencies_and_diagnostics(
        self, raw_state_np: "NDArrayLikeDict", dt: Optional[float]
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz

        s = raw_state_np["air_isentropic_density"]
        su = raw_state_np["x_momentum_isentropic"]
        u = raw_state_np["x_velocity_at_u_locations"]

        tnd_s = 1e-3 * s
        tnd_su = 300 * su
        tnd_u = np.zeros_like(s)
        tnd_u[:nx, :ny, :nz] = 50 * (
            u[:nx, :ny, :nz] + u[1 : nx + 1, :ny, :nz]
        )
        tendencies = {
            "air_isentropic_density": tnd_s,
            "x_momentum_isentropic": tnd_su,
            "x_velocity": tnd_u,
        }

        diagnostics = {"fake_variable": 2 * s}

        return tendencies, diagnostics


class FakeTendencyComponent2TestSuite(TendencyComponentTestSuite):
    @cached_property
    def component(self) -> "TendencyComponent":
        return conftest.FakeTendencyComponent2(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.storage_options,
        )

    def get_state(self) -> "DataArrayDict":
        out = self.ds.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )
        )
        out["fake_variable"] = get_dataarray_3d(
            self.hyp_data.draw(
                st_raw_field(
                    self.ds.storage_shape
                    or (self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz),
                    min_value=-1e4,
                    max_value=1e4,
                    backend=self.ds.backend,
                    storage_options=self.ds.so,
                )
            ),
            self.ds.grid,
            "kg m^-2 K^-1",
            grid_shape=(self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz),
            set_coordinates=False,
        )
        return out

    def get_validation_tendencies_and_diagnostics(
        self, raw_state_np: "NDArrayLikeDict", dt: Optional[float]
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz

        s = raw_state_np["air_isentropic_density"]
        f = raw_state_np["fake_variable"]
        v = raw_state_np["y_velocity_at_v_locations"]

        tnd_s = 0.01 * f
        tnd_sv = np.zeros_like(s)
        tnd_sv[:nx, :ny, :nz] = (
            0.5
            * s[:nx, :ny, :nz]
            * (v[:nx, :ny, :nz] + v[:nx, 1 : ny + 1, :nz])
        )
        tendencies = {
            "air_isentropic_density": tnd_s,
            "y_momentum_isentropic": tnd_sv,
        }

        return tendencies, {}
