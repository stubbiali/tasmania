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
from datetime import timedelta
from hypothesis import strategies as hyp_st
from property_cached import cached_property
from typing import Dict, Optional, TYPE_CHECKING, Tuple

from sympl._core.units import units_are_same

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.storage import deepcopy_dataarray_dict

from tests import conf
from tests.strategies import (
    st_domain,
    st_one_of,
    st_out_diagnostics,
    st_out_tendencies,
    st_overwrite_tendencies,
    st_timedeltas,
)
from tests.utilities import compare_arrays

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLike, NDArrayLikeDict

    from tasmania.python.utils.typingx import (
        Datatype,
        DiagnosticComponent,
        PairInt,
        TendencyComponent,
        TimeDelta,
    )


class DomainSuite:
    def __init__(
        self,
        hyp_data,
        backend: str,
        dtype: "Datatype",
        *,
        grid_type: Optional[str] = None,
        xaxis_length: Optional["PairInt"] = None,
        yaxis_length: Optional["PairInt"] = None,
        zaxis_length: Optional["PairInt"] = None,
        nb_min: Optional[int] = None,
        check_rebuild: bool = True
    ):
        self.hyp_data = hyp_data

        self.backend = backend
        self.backend_options = BackendOptions(
            rebuild=False, check_rebuild=check_rebuild
        )
        aligned_index = hyp_data.draw(
            st_one_of(conf.aligned_index), label="aligned_index"
        )
        self.storage_options = StorageOptions(
            dtype=dtype, aligned_index=aligned_index
        )

        nb_min = nb_min or 1
        self.nb = hyp_data.draw(
            hyp_st.integers(min_value=nb_min, max_value=max(nb_min, conf.nb)),
            label="nb",
        )
        self.domain = hyp_data.draw(
            st_domain(
                xaxis_length=xaxis_length,
                yaxis_length=yaxis_length,
                zaxis_length=zaxis_length,
                nb=self.nb,
                backend=self.backend,
                storage_options=self.storage_options,
            )
        )
        self.grid_type = (
            grid_type if grid_type in ("numerical", "physical") else "physical"
        )
        self.grid = (
            self.domain.physical_grid
            if self.grid_type == "physical"
            else self.domain.numerical_grid
        )

    @property
    def bo(self) -> BackendOptions:
        return self.backend_options

    @property
    def so(self) -> StorageOptions:
        return self.storage_options


class ComponentTestSuite(abc.ABC):
    def __init__(self, domain_suite: DomainSuite) -> None:
        self.ds = domain_suite
        self.hyp_data = self.ds.hyp_data
        set_storage_shape = self.hyp_data.draw(
            hyp_st.booleans(), label="set_storage_shape"
        )
        if set_storage_shape:
            dnx = self.hyp_data.draw(
                hyp_st.integers(min_value=1, max_value=3), label="dnx"
            )
            dny = self.hyp_data.draw(
                hyp_st.integers(min_value=1, max_value=3), label="dny"
            )
            dnz = self.hyp_data.draw(
                hyp_st.integers(min_value=1, max_value=3), label="dnz"
            )
            grid = self.ds.grid
            self.storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)
        else:
            self.storage_shape = None

    @cached_property
    @abc.abstractmethod
    def component(self):
        pass

    @abc.abstractmethod
    def get_state(self) -> "DataArrayDict":
        pass

    def assert_allclose(
        self, name: str, field_a: "NDArrayLike", field_b: "NDArrayLike"
    ) -> None:
        grid_shape = self.component.get_field_grid_shape(name)
        try:
            compare_arrays(
                field_a, field_b, slice=[slice(el) for el in grid_shape]
            )
        except AssertionError:
            raise RuntimeError(f"assert_allclose failed on {name}")


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
        raw_diagnostics_np = self.get_diagnostics(raw_state_np)

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
    def get_diagnostics(
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
            timestep = timestep or self.hyp_data.draw(
                st_timedeltas(
                    min_value=timedelta(seconds=0),
                    max_value=timedelta(seconds=3600),
                ),
                label="timestep",
            )
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
        ) = self.get_tendencies_and_diagnostics(
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
    def get_tendencies_and_diagnostics(
        self, raw_state_np: "NDArrayLikeDict", dt: Optional[float]
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        pass
