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

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity
from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils.storage_utils import empty, zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float64


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
        domain,
        smagorinsky_constant=0.18,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        smagorinsky_constant : `float`, optional
            The Smagorinsky constant. Defaults to 0.18.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `numpy.dtype`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple`, optional
            Shape of the storages.
        **kwargs :
            Keyword arguments to be directly forwarded to the parent's constructor.
        """
        super().__init__(
            domain,
            smagorinsky_constant,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            storage_shape=storage_shape,
            **kwargs
        )

        self._hv = HorizontalVelocity(
            self.grid,
            staggering=False,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            exec_info=exec_info,
            rebuild=rebuild,
        )

        self._in_u = zeros(self._storage_shape, backend, dtype, default_origin=default_origin)
        self._in_v = zeros(self._storage_shape, backend, dtype, default_origin=default_origin)
        self._out_su_tnd = zeros(self._storage_shape, backend, dtype, default_origin=default_origin)
        self._out_sv_tnd = zeros(self._storage_shape, backend, dtype, default_origin=default_origin)

    @property
    def input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

    @property
    def tendency_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()

        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]

        self._hv.get_velocity_components(in_s, in_su, in_sv, self._in_u, self._in_v)

        self._stencil(
            in_u=self._in_u,
            in_v=self._in_v,
            out_u_tnd=self._out_u_tnd,
            out_v_tnd=self._out_v_tnd,
            dx=dx,
            dy=dy,
            cs=self._cs,
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
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
