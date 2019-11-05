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

from gridtools import gtscript

# from gridtools.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


class IsentropicConservativeCoriolis(TendencyComponent):
    """
    Calculate the Coriolis forcing term for the isentropic velocity momenta.
    """

    def __init__(
        self,
        domain,
        grid_type="numerical",
        coriolis_parameter=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        managed_memory=False,
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

        coriolis_parameter : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the Coriolis
            parameter, in units compatible with [rad s^-1].
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
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
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
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._tnd_sv = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb

        self._stencil(
            in_su=state["x_momentum_isentropic"],
            in_sv=state["y_momentum_isentropic"],
            tnd_su=self._tnd_su,
            tnd_sv=self._tnd_sv,
            f=self._f,
            origin={"_all_": (nb, nb, 0)},
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
    def _stencil_defs(
        in_su: gtscript.Field[np.float64],
        in_sv: gtscript.Field[np.float64],
        tnd_su: gtscript.Field[np.float64],
        tnd_sv: gtscript.Field[np.float64],
        *,
        f: float
    ):
        with computation(PARALLEL), interval(...):
            tnd_su = f * in_sv[0, 0, 0]
            tnd_sv = -f * in_su[0, 0, 0]
