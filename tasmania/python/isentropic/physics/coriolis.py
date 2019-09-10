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
"""
This module contains:
	IsentropicConservativeCoriolis
"""
import numpy as np

import gridtools as gt
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils.storage_utils import get_storage_descriptor

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
        halo=None,
        rebuild=False,
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
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			TODO
		exec_info : `dict`, optional
			TODO
		halo : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.TendencyComponent`.
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
        descriptor = get_storage_descriptor((nx, ny, nz), dtype, halo=halo)
        self._in_su = gt.storage.zeros(descriptor, backend=backend)
        self._in_sv = gt.storage.zeros(descriptor, backend=backend)
        self._tnd_su = gt.storage.zeros(descriptor, backend=backend)
        self._tnd_sv = gt.storage.zeros(descriptor, backend=backend)

        decorator = gt.stencil(
            backend, backend_opts=backend_opts, build_info=build_info, rebuild=rebuild
        )
        self._stencil = decorator(self._stencil_defs)

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

        self._in_su.data[...] = state["x_momentum_isentropic"][...]
        self._in_sv.data[...] = state["y_momentum_isentropic"][...]

        self._stencil(
            in_su=self._in_su,
            in_sv=self._in_sv,
            tnd_su=self._tnd_su,
            tnd_sv=self._tnd_sv,
            f=self._f,
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

        tendencies = {
            "x_momentum_isentropic": self._tnd_su.data,
            "y_momentum_isentropic": self._tnd_sv.data,
        }

        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_su: gt.storage.f64_sd,
        in_sv: gt.storage.f64_sd,
        tnd_su: gt.storage.f64_sd,
        tnd_sv: gt.storage.f64_sd,
        *,
        f: float
    ):
        tnd_su = f * in_sv[0, 0, 0]
        tnd_sv = -f * in_su[0, 0, 0]
