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
from typing import Dict, Optional, Sequence, TYPE_CHECKING

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.core_components import TendencyComponent

if TYPE_CHECKING:
    from sympl import DataArray
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicHorizontalDiffusion(TendencyComponent):
    """
    Calculate the tendencies due to horizontal diffusion for the
    prognostic fields of an isentropic model state. The class is
    always instantiated over the numerical grid of the
    underlying domain.
    """

    def __init__(
        self,
        domain: "Domain",
        diffusion_type: str,
        diffusion_coeff: "DataArray",
        diffusion_coeff_max: "DataArray",
        diffusion_damp_depth: int,
        moist: bool = False,
        diffusion_moist_coeff: Optional["DataArray"] = None,
        diffusion_moist_coeff_max: Optional["DataArray"] = None,
        diffusion_moist_damp_depth: Optional[int] = None,
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
        diffusion_type : str
            The type of numerical diffusion to implement.
            See :class:`~tasmania.HorizontalDiffusion` for all available options.
        diffusion_coeff : sympl.DataArray
            1-item array representing the diffusion coefficient;
            in units compatible with [s^-1].
        diffusion_coeff_max : sympl.DataArray
            1-item array representing the maximum value assumed by the
            diffusion coefficient close to the upper boundary;
            in units compatible with [s^-1].
        diffusion_damp_depth : int
            Depth of the damping region.
        moist : `bool`, optional
            ``True`` if water species are included in the model and should
            be diffused, ``False`` otherwise. Defaults to ``False``.
        diffusion_moist_coeff : `sympl.DataArray`, optional
            1-item array representing the diffusion coefficient for the
            water species; in units compatible with [s^-1].
        diffusion_moist_coeff_max : `sympl.DataArray`, optional
            1-item array representing the maximum value assumed by the
            diffusion coefficient for the water species close to the upper
            boundary; in units compatible with [s^-1].
        diffusion_damp_depth : int
            Depth of the damping region for the water species.
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
        self._moist = moist and diffusion_moist_coeff is not None

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

        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        nb = self.horizontal_boundary.nb

        diff_coeff = diffusion_coeff.to_units("s^-1").values.item()
        diff_coeff_max = diffusion_coeff_max.to_units("s^-1").values.item()

        self._core = HorizontalDiffusion.factory(
            diffusion_type,
            self.storage_shape,
            dx,
            dy,
            diff_coeff,
            diff_coeff_max,
            diffusion_damp_depth,
            nb,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_options=self.storage_options,
        )

        if self._moist:
            diff_moist_coeff = diffusion_moist_coeff.to_units(
                "s^-1"
            ).values.item()
            diff_moist_coeff_max = (
                diff_moist_coeff
                if diffusion_moist_coeff_max is None
                else diffusion_moist_coeff_max.to_units("s^-1").values.item()
            )
            diff_moist_damp_depth = (
                0
                if diffusion_moist_damp_depth is None
                else diffusion_moist_damp_depth
            )

            self._core_moist = HorizontalDiffusion.factory(
                diffusion_type,
                self.storage_shape,
                dx,
                dy,
                diff_moist_coeff,
                diff_moist_coeff_max,
                diff_moist_damp_depth,
                nb,
                backend=backend,
                backend_options=self.backend_options,
                storage_options=self.storage_options,
            )
        else:
            self._core_moist = None

    @property
    def input_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
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

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1 s^-1"}

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
        self._core(
            state["air_isentropic_density"],
            out_tendencies["air_isentropic_density"],
            overwrite_output=overwrite_tendencies["air_isentropic_density"],
        )
        self._core(
            state["x_momentum_isentropic"],
            out_tendencies["x_momentum_isentropic"],
            overwrite_output=overwrite_tendencies["x_momentum_isentropic"],
        )
        self._core(
            state["y_momentum_isentropic"],
            out_tendencies["y_momentum_isentropic"],
            overwrite_output=overwrite_tendencies["y_momentum_isentropic"],
        )

        if self._moist:
            self._core_moist(
                state[mfwv],
                out_tendencies[mfwv],
                overwrite_output=overwrite_tendencies[mfwv],
            )
            self._core_moist(
                state[mfcw],
                out_tendencies[mfcw],
                overwrite_output=overwrite_tendencies[mfcw],
            )
            self._core_moist(
                state[mfpw],
                out_tendencies[mfpw],
                overwrite_output=overwrite_tendencies[mfpw],
            )
