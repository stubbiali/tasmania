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
from typing import Optional, TYPE_CHECKING, Tuple

from gt4py import gtscript

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


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
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
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
        smagorinsky_constant : `float`, optional
            The Smagorinsky constant. Defaults to 0.18.
        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            class.
        """
        super().__init__(domain, "numerical", **kwargs)

        self._cs = smagorinsky_constant
        self._exec_info = exec_info

        assert (
            self.horizontal_boundary.nb >= 2
        ), "The number of boundary layers must be greater or equal than two."

        self._nb = max(2, self.horizontal_boundary.nb)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = (
            (nx, ny, nz) if storage_shape is None else storage_shape
        )
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz, error_msg
        self._storage_shape = storage_shape

        self._out_u_tnd = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_v_tnd = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        if is_gt(backend):
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.array_dict_t
    ) -> Tuple[taz_types.array_dict_t, taz_types.array_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()

        self._stencil(
            in_u=state["x_velocity"],
            in_v=state["y_velocity"],
            out_u_tnd=self._out_u_tnd,
            out_v_tnd=self._out_v_tnd,
            dx=dx,
            dy=dy,
            cs=self._cs,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
            validate_args=True,
        )

        tendencies = {
            "x_velocity": self._out_u_tnd,
            "y_velocity": self._out_v_tnd,
        }
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_numpy(
        in_u: np.ndarray,
        in_v: np.ndarray,
        out_u_tnd: np.ndarray,
        out_v_tnd: np.ndarray,
        *,
        dx: float,
        dy: float,
        cs: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        s00 = (
            in_u[ib : ie + 2, jb - 1 : je + 1, k]
            - in_u[ib - 2 : ie, jb - 1 : je + 1, k]
        ) / (2.0 * dx)
        s01 = 0.5 * (
            (
                in_u[ib - 1 : ie + 1, jb : je + 2, k]
                - in_u[ib - 1 : ie + 1, jb - 2 : je, k]
            )
            / (2.0 * dy)
            + (
                in_v[ib : ie + 2, jb - 1 : je + 1, k]
                - in_v[ib - 2 : ie, jb - 1 : je + 1, k]
            )
            / (2.0 * dx)
        )
        s11 = (
            in_v[ib - 1 : ie + 1, jb : je + 2, k]
            - in_v[ib - 1 : ie + 1, jb - 2 : je, k]
        ) / (2.0 * dy)
        nu = (
            cs ** 2
            * dx
            * dy
            * (2.0 * (s00 ** 2 + 2.0 * s01 ** 2 + s11 ** 2)) ** 0.5
        )
        out_u_tnd[ib:ie, jb:je, k] = 2.0 * (
            (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2])
            / (2.0 * dy)
        )
        out_v_tnd[ib:ie, jb:je, k] = 2.0 * (
            (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2])
            / (2.0 * dy)
        )

    @staticmethod
    def _stencil_gt_defs(
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_u_tnd: gtscript.Field["dtype"],
        out_v_tnd: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        cs: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            s00 = (in_u[+1, 0, 0] - in_u[-1, 0, 0]) / (2.0 * dx)
            s01 = 0.5 * (
                (in_u[0, +1, 0] - in_u[0, -1, 0]) / (2.0 * dy)
                + (in_v[+1, 0, 0] - in_v[-1, 0, 0]) / (2.0 * dx)
            )
            s11 = (in_v[0, +1, 0] - in_v[0, -1, 0]) / (2.0 * dy)
            nu = (
                cs ** 2
                * dx
                * dy
                * (2.0 * (s00 ** 2 + 2.0 * s01 ** 2 + s11 ** 2)) ** 0.5
            )
            out_u_tnd = 2.0 * (
                (nu[+1, 0, 0] * s00[+1, 0, 0] - nu[-1, 0, 0] * s00[-1, 0, 0])
                / (2.0 * dx)
                + (nu[0, +1, 0] * s01[0, +1, 0] - nu[0, -1, 0] * s01[0, -1, 0])
                / (2.0 * dy)
            )
            out_v_tnd = 2.0 * (
                (nu[+1, 0, 0] * s01[+1, 0, 0] - nu[-1, 0, 0] * s01[-1, 0, 0])
                / (2.0 * dx)
                + (nu[0, +1, 0] * s11[0, +1, 0] - nu[0, -1, 0] * s11[0, -1, 0])
                / (2.0 * dy)
            )
