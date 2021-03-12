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
from typing import Optional, Sequence, TYPE_CHECKING, Tuple

from gt4py import gtscript

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.utils import typing as ty

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class IsentropicConservativeCoriolis(TendencyComponent):
    """
    Calculate the Coriolis forcing term for the isentropic velocity momenta.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        coriolis_parameter: Optional[DataArray] = None,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
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
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        **kwargs :
            Keyword arguments to be directly forwarded to the parent's
            constructor.
        """
        super().__init__(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
            **kwargs
        )

        self._nb = (
            self.horizontal_boundary.nb if grid_type == "numerical" else 0
        )
        self._f = (
            coriolis_parameter.to_units("rad s^-1").values.item()
            if coriolis_parameter is not None
            else 1e-4
        )

        storage_shape = self.get_storage_shape(storage_shape)
        self._tnd_su = self.zeros(shape=storage_shape)
        self._tnd_sv = self.zeros(shape=storage_shape)

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile("coriolis")

    @property
    def input_properties(self) -> ty.PropertiesDict:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
        }

        return return_dict

    @property
    def tendency_properties(self) -> ty.PropertiesDict:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> ty.PropertiesDict:
        return {}

    def array_call(
        self, state: ty.StorageDict
    ) -> Tuple[ty.StorageDict, ty.StorageDict]:
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
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

        tendencies = {
            "x_momentum_isentropic": self._tnd_su,
            "y_momentum_isentropic": self._tnd_sv,
        }
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="coriolis")
    def _stencil_numpy(
        in_su: np.ndarray,
        in_sv: np.ndarray,
        tnd_su: np.ndarray,
        tnd_sv: np.ndarray,
        *,
        f: float,
        origin: ty.TripletInt,
        domain: ty.TripletInt
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        tnd_su[i, j, k] = f * in_sv[i, j, k]
        tnd_sv[i, j, k] = -f * in_su[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="coriolis")
    def _stencil_gt4py(
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        tnd_su: gtscript.Field["dtype"],
        tnd_sv: gtscript.Field["dtype"],
        *,
        f: "dtype"
    ) -> None:
        with computation(PARALLEL), interval(...):
            tnd_su = f * in_sv
            tnd_sv = -f * in_su
