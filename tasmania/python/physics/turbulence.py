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
from typing import Optional, Sequence, TYPE_CHECKING

from sympl._core.time import Timer

from gt4py import gtscript

from tasmania.python.framework.core_components import TendencyComponent
from tasmania.python.framework.tag import (
    stencil_definition,
    subroutine_definition,
)

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TripletInt


class Smagorinsky2d(TendencyComponent):
    """
    Implementation of the Smagorinsky turbulence model for a
    two-dimensional flow.
    The class is instantiated over the *numerical* grid of the
    underlying domain.

    References
    ----------
    Rösler, M. (2015). *The Smagorinsky turbulence model.* Master thesis, \
        Freie Universität Berlin.
    """

    def __init__(
        self,
        domain: "Domain",
        smagorinsky_constant: float = 0.18,
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
        smagorinsky_constant : `float`, optional
            The Smagorinsky constant. Defaults to 0.18.
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
            Additional keyword arguments to be directly forwarded to the parent
            class.
        """
        super().__init__(
            domain,
            "numerical",
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
            **kwargs
        )

        self._cs = smagorinsky_constant

        assert (
            self.horizontal_boundary.nb >= 2
        ), "The number of boundary layers must be greater or equal than two."

        self._nb = max(2, self.horizontal_boundary.nb)

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "core": self.get_subroutine_definition("smagorinsky_core"),
            "set_output": self.get_subroutine_definition("set_output"),
        }
        self._stencil = self.compile_stencil("smagorinsky")

    @property
    def input_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def diagnostic_properties(self) -> "PropertyDict":
        return {}

    def array_call(
        self,
        state: "NDArrayLikeDict",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: "NDArrayLikeDict",
    ):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        Timer.start(label="stencil")
        self._stencil(
            in_u=state["x_velocity"],
            in_v=state["y_velocity"],
            out_u_tnd=out_tendencies["x_velocity"],
            out_v_tnd=out_tendencies["y_velocity"],
            dx=dx,
            dy=dy,
            cs=self._cs,
            ow_out_u_tnd=overwrite_tendencies["x_velocity"],
            ow_out_v_tnd=overwrite_tendencies["y_velocity"],
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="smagorinsky"
    )
    def _stencil_numpy(
        in_u: np.ndarray,
        in_v: np.ndarray,
        out_u_tnd: np.ndarray,
        out_v_tnd: np.ndarray,
        *,
        dx: float,
        dy: float,
        cs: float,
        ow_out_u_tnd: bool,
        ow_out_v_tnd: bool,
        origin: "TripletInt",
        domain: "TripletInt"
    ) -> None:
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        tmp_out_u_tnd, tmp_out_v_tnd = core(
            in_u, in_v, dx, dy, cs, ib, ie, jb, je, k
        )
        set_output(out_u_tnd[ib:ie, jb:je, k], tmp_out_u_tnd, ow_out_u_tnd)
        set_output(out_v_tnd[ib:ie, jb:je, k], tmp_out_v_tnd, ow_out_v_tnd)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smagorinsky")
    def _stencil_gt4py(
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_u_tnd: gtscript.Field["dtype"],
        out_v_tnd: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        cs: float,
        ow_out_u_tnd: bool,
        ow_out_v_tnd: bool
    ) -> None:
        from __externals__ import core, set_output

        with computation(PARALLEL), interval(...):
            tmp_out_u_tnd, tmp_out_v_tnd = core(in_u, in_v, dx, dy, cs)
            out_u_tnd = set_output(out_u_tnd, tmp_out_u_tnd, ow_out_u_tnd)
            out_v_tnd = set_output(out_v_tnd, tmp_out_v_tnd, ow_out_v_tnd)

    @staticmethod
    @subroutine_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"),
        stencil="smagorinsky_core",
    )
    def _core_numpy(u, v, dx, dy, cs, ib, ie, jb, je, k):
        s00 = (
            u[ib : ie + 2, jb - 1 : je + 1, k]
            - u[ib - 2 : ie, jb - 1 : je + 1, k]
        ) / (2.0 * dx)
        s01 = 0.5 * (
            (
                u[ib - 1 : ie + 1, jb : je + 2, k]
                - u[ib - 1 : ie + 1, jb - 2 : je, k]
            )
            / (2.0 * dy)
            + (
                v[ib : ie + 2, jb - 1 : je + 1, k]
                - v[ib - 2 : ie, jb - 1 : je + 1, k]
            )
            / (2.0 * dx)
        )
        s11 = (
            v[ib - 1 : ie + 1, jb : je + 2, k]
            - v[ib - 1 : ie + 1, jb - 2 : je, k]
        ) / (2.0 * dy)
        nu = (
            cs ** 2
            * dx
            * dy
            * (2.0 * (s00 ** 2 + 2.0 * s01 ** 2 + s11 ** 2)) ** 0.5
        )
        u_tnd = 2.0 * (
            (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2])
            / (2.0 * dy)
        )
        v_tnd = 2.0 * (
            (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2])
            / (2.0 * dy)
        )
        return u_tnd, v_tnd

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="smagorinsky_core")
    @gtscript.function
    def _core_gt4py(u, v, dx, dy, cs):
        s00 = (u[+1, 0, 0] - u[-1, 0, 0]) / (2.0 * dx)
        s01 = 0.5 * (
            (u[0, +1, 0] - u[0, -1, 0]) / (2.0 * dy)
            + (v[+1, 0, 0] - v[-1, 0, 0]) / (2.0 * dx)
        )
        s11 = (v[0, +1, 0] - v[0, -1, 0]) / (2.0 * dy)
        nu = (
            cs ** 2
            * dx
            * dy
            * (2.0 * (s00 ** 2 + 2.0 * s01 ** 2 + s11 ** 2)) ** 0.5
        )
        u_tnd = 2.0 * (
            (nu[+1, 0, 0] * s00[+1, 0, 0] - nu[-1, 0, 0] * s00[-1, 0, 0])
            / (2.0 * dx)
            + (nu[0, +1, 0] * s01[0, +1, 0] - nu[0, -1, 0] * s01[0, -1, 0])
            / (2.0 * dy)
        )
        v_tnd = 2.0 * (
            (nu[+1, 0, 0] * s01[+1, 0, 0] - nu[-1, 0, 0] * s01[-1, 0, 0])
            / (2.0 * dx)
            + (nu[0, +1, 0] * s11[0, +1, 0] - nu[0, -1, 0] * s11[0, -1, 0])
            / (2.0 * dy)
        )
        return u_tnd, v_tnd
