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
from typing import Optional, Tuple

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.grids.domain import Domain
from tasmania.python.utils import types
from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float64


class BurgersHorizontalDiffusion(TendencyComponent):
    """
    A :class:`tasmania.TendencyComponent` calculating the tendencies
    due to diffusion for the 2-D Burgers equations.
    """

    def __init__(
        self,
        domain: Domain,
        grid_type: str,
        diffusion_type: str,
        diffusion_coeff: DataArray,
        *,
        backend: str = "numpy",
        backend_opts: Optional[types.options_dict_t] = None,
        build_info: Optional[types.options_dict_t] = None,
        dtype: types.dtype_t = datatype,
        exec_info: Optional[types.mutable_options_dict_t] = None,
        default_origin: Optional[types.triplet_int_t] = None,
        rebuild: bool = False,
        managed_memory: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : str
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical'.

        diffusion_type : str
            String specifying the desired type of horizontal diffusion.
            See :class:`tasmania.HorizontalDiffusion` for all available options.
        diffusion_coeff : sympl.DataArray
            1-item :class:`sympl.DataArray` representing the diffusion
            coefficient. The units should be compatible with 'm^2 s^-1'.
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
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        kwargs :
            Keyword arguments to be broadcast to :class:`sympl.TendencyComponent`.
        """
        super().__init__(domain, grid_type, **kwargs)

        nx, ny = self.grid.grid_xy.nx, self.grid.grid_xy.ny
        dx = self.grid.grid_xy.dx.to_units("m").values.item()
        dy = self.grid.grid_xy.dy.to_units("m").values.item()

        self._diffuser = HorizontalDiffusion.factory(
            diffusion_type,
            (nx, ny, 1),
            dx,
            dy,
            diffusion_coeff=diffusion_coeff.to_units("m^2 s^-1").values.item(),
            diffusion_coeff_max=diffusion_coeff.to_units("m^2 s^-1").values.item(),
            diffusion_damp_depth=0,
            nb=self.horizontal_boundary.nb,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            managed_memory=managed_memory,
        )

        self._out_u_tnd = zeros(
            (nx, ny, 1),
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_v_tnd = zeros(
            (nx, ny, 1),
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

    @property
    def input_properties(self) -> types.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> types.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def diagnostic_properties(self) -> types.properties_dict_t:
        return {}

    def array_call(
        self, state: types.gtstorage_dict_t
    ) -> Tuple[types.gtstorage_dict_t, types.gtstorage_dict_t]:
        self._diffuser(state["x_velocity"], self._out_u_tnd)
        self._diffuser(state["y_velocity"], self._out_v_tnd)

        tendencies = {"x_velocity": self._out_u_tnd, "y_velocity": self._out_v_tnd}
        diagnostics = {}

        return tendencies, diagnostics
