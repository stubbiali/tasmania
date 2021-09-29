# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from typing import Dict, Optional, TYPE_CHECKING

from sympl import DataArray

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.core_components import TendencyComponent

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

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
        enable_checks: bool = True,
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
        enable_checks : `bool`, optional
            TODO
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
            enable_checks=enable_checks,
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

    @property
    def input_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
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
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        self._diffuser(
            state["x_velocity"],
            out_tendencies["x_velocity"],
            overwrite_output=overwrite_tendencies["x_velocity"],
        )
        self._diffuser(
            state["y_velocity"],
            out_tendencies["y_velocity"],
            overwrite_output=overwrite_tendencies["y_velocity"],
        )
