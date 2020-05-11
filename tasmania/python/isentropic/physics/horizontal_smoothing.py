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
from typing import Optional, TYPE_CHECKING

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import get_storage_shape, zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


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
        gt_powered: bool = None,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        smooth_type : str
            The type of numerical smoothing to implement.
            See :class:`~tasmania.HorizontalSmoothing` for all available options.
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
        gt_powered : `bool`, optional
            TODO
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
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        """
        self._moist = moist and smooth_moist_coeff is not None

        super().__init__(domain, "numerical")

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self.horizontal_boundary.nb

        shape = get_storage_shape(storage_shape, min_shape=(nx + 1, ny + 1, nz + 1))

        self._core = HorizontalSmoothing.factory(
            smooth_type,
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            managed_memory=managed_memory,
        )

        if self._moist:
            smooth_moist_coeff_max = (
                smooth_moist_coeff
                if smooth_moist_coeff_max is None
                else smooth_moist_coeff_max
            )
            smooth_moist_damp_depth = (
                0 if smooth_moist_damp_depth is None else smooth_moist_damp_depth
            )

            self._core_moist = HorizontalSmoothing.factory(
                smooth_type,
                shape,
                smooth_moist_coeff,
                smooth_moist_coeff_max,
                smooth_moist_damp_depth,
                nb,
                gt_powered=gt_powered,
                backend=backend,
                backend_opts=backend_opts,
                build_info=build_info,
                dtype=dtype,
                exec_info=exec_info,
                default_origin=default_origin,
                rebuild=rebuild,
                managed_memory=managed_memory,
            )
        else:
            self._core_moist = None

        self._out_s = zeros(
            shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_su = zeros(
            shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_sv = zeros(
            shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        if self._moist:
            self._out_qv = zeros(
                shape,
                gt_powered=gt_powered,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )
            self._out_qc = zeros(
                shape,
                gt_powered=gt_powered,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )
            self._out_qr = zeros(
                shape,
                gt_powered=gt_powered,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return self.input_properties

    def array_call(self, state: taz_types.array_dict_t) -> taz_types.array_dict_t:
        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]
        self._core(in_s, self._out_s)
        self._core(in_su, self._out_su)
        self._core(in_sv, self._out_sv)

        return_dict = {
            "air_isentropic_density": self._out_s,
            "x_momentum_isentropic": self._out_su,
            "y_momentum_isentropic": self._out_sv,
        }

        if self._moist:
            in_qv = state[mfwv]
            in_qc = state[mfcw]
            in_qr = state[mfpw]

            self._core_moist(in_qv, self._out_qv)
            self._core_moist(in_qc, self._out_qc)
            self._core_moist(in_qr, self._out_qr)

            return_dict[mfwv] = self._out_qv
            return_dict[mfcw] = self._out_qc
            return_dict[mfpw] = self._out_qr

        return return_dict
