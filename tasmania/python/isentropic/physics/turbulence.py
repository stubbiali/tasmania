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

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity
from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


class IsentropicSmagorinsky(Smagorinsky2d):
    """
    Implementation of the Smagorinsky turbulence model for the
    isentropic model. The conservative form of the governing
    equations is used.
    The class is instantiated over the *numerical* grid of the
    underlying domain.
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
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
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
            Keyword arguments to be directly forwarded to the parent's
            constructor.
        """
        super().__init__(
            domain,
            smagorinsky_constant,
            backend=backend,
            backend_opts=backend_opts,
            dtype=dtype,
            build_info=build_info,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            storage_shape=storage_shape,
            managed_memory=managed_memory,
            **kwargs
        )

        self._hv = HorizontalVelocity(
            self.grid,
            staggering=False,
            backend=backend,
            backend_opts=backend_opts,
            dtype=dtype,
            build_info=build_info,
            exec_info=exec_info,
            rebuild=rebuild,
        )

        self._in_u = zeros(
            self._storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._in_v = zeros(
            self._storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_su_tnd = zeros(
            self._storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_sv_tnd = zeros(
            self._storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
        }

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
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

        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]

        self._hv.get_velocity_components(
            in_s, in_su, in_sv, self._in_u, self._in_v
        )

        self._stencil(
            in_u=self._in_u,
            in_v=self._in_v,
            out_u_tnd=self._out_u_tnd,
            out_v_tnd=self._out_v_tnd,
            dx=dx,
            dy=dy,
            cs=self._cs,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
            validate_args=False,
        )

        self._hv.get_momenta(
            in_s,
            self._out_u_tnd,
            self._out_v_tnd,
            self._out_su_tnd,
            self._out_sv_tnd,
        )

        tendencies = {
            "x_momentum_isentropic": self._out_su_tnd,
            "y_momentum_isentropic": self._out_sv_tnd,
        }
        diagnostics = {}

        return tendencies, diagnostics
