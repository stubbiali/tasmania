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
from typing import Optional, Sequence, TYPE_CHECKING

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.core_components import DiagnosticComponent

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicHorizontalSmoothing(DiagnosticComponent):
    """
    Apply numerical smoothing to the prognostic fields of an
    isentropic model state. The class is always instantiated
    over the numerical grid of the underlying domain.
    """

    def __init__(
        self,
        domain: "Domain",
        smooth_type: str,
        smooth_coeff: float,
        smooth_coeff_max: float,
        smooth_damp_depth: int,
        moist: bool = False,
        smooth_moist_coeff: Optional[float] = None,
        smooth_moist_coeff_max: Optional[float] = None,
        smooth_moist_damp_depth: Optional[int] = None,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        smooth_type : str
            The type of numerical smoothing to implement.
            See :class:`~tasmania.HorizontalSmoothing` for all available
            options.
        smooth_coeff : float
            The smoothing coefficient.
        smooth_coeff_max : float
            The maximum value assumed by the smoothing coefficient close to the
            upper boundary.
        smooth_damp_depth : int
            Depth of the damping region.
        moist : `bool`, optional
            ``True`` if water species are included in the model and should
            be smoothed, ``False`` otherwise. Defaults to ``False``.
        smooth_moist_coeff : `float`, optional
            The smoothing coefficient for the water species.
        smooth_moist_coeff_max : `float`, optional
            The maximum value assumed by the smoothing coefficient for the water
            species close to the upper boundary.
        smooth_damp_depth : int
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
        """
        self._moist = moist and smooth_moist_coeff is not None

        super().__init__(
            domain,
            "numerical",
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

        self._core = HorizontalSmoothing.factory(
            smooth_type,
            self.storage_shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            self.horizontal_boundary.nb,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_options=self.storage_options,
        )

        if self._moist:
            smooth_moist_coeff_max = (
                smooth_moist_coeff if smooth_moist_coeff_max is None else smooth_moist_coeff_max
            )
            smooth_moist_damp_depth = (
                0 if smooth_moist_damp_depth is None else smooth_moist_damp_depth
            )
            self._core_moist = HorizontalSmoothing.factory(
                smooth_type,
                self.storage_shape,
                smooth_moist_coeff,
                smooth_moist_coeff_max,
                smooth_moist_damp_depth,
                self.horizontal_boundary.nb,
                backend=self.backend,
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
    def diagnostic_properties(self) -> "PropertyDict":
        return self.input_properties

    def array_call(self, state: "NDArrayLikeDict", out: "NDArrayLikeDict") -> None:
        self._core(state["air_isentropic_density"], out["air_isentropic_density"])
        self._core(state["x_momentum_isentropic"], out["x_momentum_isentropic"])
        self._core(state["y_momentum_isentropic"], out["y_momentum_isentropic"])

        if self._moist:
            self._core_moist(state[mfwv], out[mfwv])
            self._core_moist(state[mfcw], out[mfcw])
            self._core_moist(state[mfpw], out[mfpw])
