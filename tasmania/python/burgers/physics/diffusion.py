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
from sympl import DataArray
from typing import Optional, TYPE_CHECKING, Tuple

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils import typing

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class BurgersHorizontalDiffusion(TendencyComponent):
    """
    A :class:`tasmania.TendencyComponent` calculating the tendencies
    due to diffusion for the 2-D Burgers equations.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        diffusion_type: str,
        diffusion_coeff: DataArray,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : str
            The type of grid over which instantiating the class.
            Either "physical" or "numerical".
        diffusion_type : str
            String specifying the desired type of horizontal diffusion.
            See :class:`tasmania.HorizontalDiffusion` for all available options.
        diffusion_coeff : sympl.DataArray
            1-item :class:`sympl.DataArray` representing the diffusion
            coefficient. The units should be compatible with 'm^2 s^-1'.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        kwargs :
            Keyword arguments to be broadcast to :class:`sympl.TendencyComponent`.
        """
        super().__init__(
            domain,
            grid_type,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
            **kwargs
        )

        nx, ny = self.grid.grid_xy.nx, self.grid.grid_xy.ny
        dx = self.grid.grid_xy.dx.to_units("m").values.item()
        dy = self.grid.grid_xy.dy.to_units("m").values.item()

        self._diffuser = HorizontalDiffusion.factory(
            diffusion_type,
            (nx, ny, 1),
            dx,
            dy,
            diffusion_coeff=diffusion_coeff.to_units("m^2 s^-1").values.item(),
            diffusion_coeff_max=diffusion_coeff.to_units(
                "m^2 s^-1"
            ).values.item(),
            diffusion_damp_depth=0,
            nb=self.horizontal_boundary.nb,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        self._out_u_tnd = self.zeros(shape=(nx, ny, 1))
        self._out_v_tnd = self.zeros(shape=(nx, ny, 1))

    @property
    def input_properties(self) -> typing.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> typing.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def diagnostic_properties(self) -> typing.properties_dict_t:
        return {}

    def array_call(
        self, state: typing.array_dict_t
    ) -> Tuple[typing.array_dict_t, typing.array_dict_t]:
        self._diffuser(state["x_velocity"], self._out_u_tnd)
        self._diffuser(state["y_velocity"], self._out_v_tnd)

        tendencies = {
            "x_velocity": self._out_u_tnd,
            "y_velocity": self._out_v_tnd,
        }
        diagnostics = {}

        return tendencies, diagnostics
