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
from hypothesis import strategies as hyp_st
from typing import Optional, TYPE_CHECKING

from tasmania import BackendOptions, StorageOptions

from tests import conf
from tests.strategies import st_one_of, st_domain

if TYPE_CHECKING:
    from tasmania.python.utils.typingx import Datatype, PairInt


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
        check_rebuild: bool = True,
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
            grid = self.grid
            self.storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)
        else:
            self.storage_shape = None

    @property
    def bo(self) -> BackendOptions:
        return self.backend_options

    @property
    def so(self) -> StorageOptions:
        return self.storage_options
