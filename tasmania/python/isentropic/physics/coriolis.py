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
import numpy as np
from sympl import DataArray
from typing import Optional, TYPE_CHECKING, Tuple

from gt4py import gtscript

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


class IsentropicConservativeCoriolis(TendencyComponent):
    """
    Calculate the Coriolis forcing term for the isentropic velocity momenta.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        coriolis_parameter: Optional[DataArray] = None,
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" and "numerical" (default).
        coriolis_parameter : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the Coriolis
            parameter, in units compatible with [rad s^-1].
        gt_powered : `bool`, optional
            TODO
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.
        **kwargs :
            Keyword arguments to be directly forwarded to the parent's constructor.
        """
        super().__init__(domain, grid_type, **kwargs)

        self._nb = self.horizontal_boundary.nb if grid_type == "numerical" else 0
        self._exec_info = exec_info

        self._f = (
            coriolis_parameter.to_units("rad s^-1").values.item()
            if coriolis_parameter is not None
            else 1e-4
        )

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = (nx, ny, nz) if storage_shape is None else storage_shape
        error_msg = "storage_shape must be larger or equal than {}.".format((nx, ny, nz))
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz, error_msg

        self._tnd_su = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._tnd_sv = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        if gt_powered:
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=backend,
                build_info=build_info,
                dtypes={"dtype": dtype},
                rebuild=rebuild,
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.array_dict_t
    ) -> Tuple[taz_types.array_dict_t, taz_types.array_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb

        self._stencil(
            in_su=state["x_momentum_isentropic"],
            in_sv=state["y_momentum_isentropic"],
            tnd_su=self._tnd_su,
            tnd_sv=self._tnd_sv,
            f=self._f,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

        tendencies = {
            "x_momentum_isentropic": self._tnd_su,
            "y_momentum_isentropic": self._tnd_sv,
        }

        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_numpy(
        in_su: np.ndarray,
        in_sv: np.ndarray,
        tnd_su: np.ndarray,
        tnd_sv: np.ndarray,
        *,
        f: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        tnd_su[i, j, k] = f * in_sv[i, j, k]
        tnd_sv[i, j, k] = -f * in_su[i, j, k]

    @staticmethod
    def _stencil_gt_defs(
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        tnd_su: gtscript.Field["dtype"],
        tnd_sv: gtscript.Field["dtype"],
        *,
        f: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            tnd_su = f * in_sv
            tnd_sv = -f * in_su
