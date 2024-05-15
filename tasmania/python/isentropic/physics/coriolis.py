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
import numpy as np
from typing import Dict, Optional, Sequence, TYPE_CHECKING

from sympl._core.data_array import DataArray
from sympl._core.time import Timer

from gt4py import gtscript

from tasmania.python.framework.core_components import TendencyComponent
from tasmania.python.framework.tag import stencil_definition

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TripletInt


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
        enable_checks: bool = True,
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
        enable_checks : `bool`, optional
            TODO
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
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
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

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "set_output": self.get_subroutine_definition("set_output")
        }
        self._stencil = self.compile_stencil("coriolis")

    @property
    def input_properties(self) -> "PropertyDict":
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
    def tendency_properties(self) -> "PropertyDict":
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
    def diagnostic_properties(self) -> "PropertyDict":
        return {}

    def array_call(
        self,
        state: "NDArrayLikeDict",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb
        Timer.start(label="stencil")
        self._stencil(
            in_su=state["x_momentum_isentropic"],
            in_sv=state["y_momentum_isentropic"],
            tnd_su=out_tendencies["x_momentum_isentropic"],
            tnd_sv=out_tendencies["y_momentum_isentropic"],
            f=self._f,
            ow_tnd_su=overwrite_tendencies["x_momentum_isentropic"],
            ow_tnd_sv=overwrite_tendencies["y_momentum_isentropic"],
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="coriolis")
    def _stencil_numpy(
        in_su: np.ndarray,
        in_sv: np.ndarray,
        tnd_su: np.ndarray,
        tnd_sv: np.ndarray,
        *,
        f: float,
        ow_tnd_su: bool,
        ow_tnd_sv: bool,
        origin: "TripletInt",
        domain: "TripletInt"
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        tmp_tnd_su = f * in_sv[i, j, k]
        set_output(tnd_su[i, j, k], tmp_tnd_su, ow_tnd_su)
        tmp_tnd_sv = -f * in_su[i, j, k]
        set_output(tnd_sv[i, j, k], tmp_tnd_sv, ow_tnd_sv)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="coriolis")
    def _stencil_gt4py(
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        tnd_su: gtscript.Field["dtype"],
        tnd_sv: gtscript.Field["dtype"],
        *,
        f: "dtype",
        ow_tnd_su: bool,
        ow_tnd_sv: bool
    ) -> None:
        from __externals__ import set_output

        with computation(PARALLEL), interval(...):
            tmp_tnd_su = f * in_sv
            tnd_su = set_output(tnd_su, tmp_tnd_su, ow_tnd_su)
            tmp_tnd_sv = -f * in_su
            tnd_sv = set_output(tnd_sv, tmp_tnd_sv, ow_tnd_sv)
