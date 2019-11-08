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

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils.storage_utils import get_storage_shape, zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float64


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
        domain,
        diffusion_type,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        moist=False,
        diffusion_moist_coeff=None,
        diffusion_moist_coeff_max=None,
        diffusion_moist_damp_depth=None,
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
            `True` if water species are included in the model and should
            be diffused, `False` otherwise. Defaults to `False`.
        diffusion_moist_coeff : `sympl.DataArray`, optional
            1-item array representing the diffusion coefficient for the
            water species; in units compatible with [s^-1].
        diffusion_moist_coeff_max : `sympl.DataArray`, optional
            1-item array representing the maximum value assumed by the
            diffusion coefficient for the water species close to the upper boundary;
            in units compatible with [s^-1].
        diffusion_damp_depth : int
            Depth of the damping region for the water species.
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
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        **kwargs :
            Keyword arguments to be directly forwarded to the parent's constructor.
        """
        self._moist = moist and diffusion_moist_coeff is not None

        super().__init__(domain, "numerical", **kwargs)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        nb = self.horizontal_boundary.nb

        diff_coeff = diffusion_coeff.to_units("s^-1").values.item()
        diff_coeff_max = diffusion_coeff_max.to_units("s^-1").values.item()

        shape = storage_shape or (nx + 1, ny + 1, nz + 1)
        shape = get_storage_shape(shape, min_shape=(nx + 1, ny + 1, nz + 1))

        self._core = HorizontalDiffusion.factory(
            diffusion_type,
            shape,
            dx,
            dy,
            diff_coeff,
            diff_coeff_max,
            diffusion_damp_depth,
            nb,
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
            diff_moist_coeff = diffusion_moist_coeff.to_units("s^-1").values.item()
            diff_moist_coeff_max = (
                diff_moist_coeff
                if diffusion_moist_coeff_max is None
                else diffusion_moist_coeff_max.to_units("s^-1").values.item()
            )
            diff_moist_damp_depth = (
                0 if diffusion_moist_damp_depth is None else diffusion_moist_damp_depth
            )

            self._core_moist = HorizontalDiffusion.factory(
                diffusion_type,
                shape,
                dx,
                dy,
                diff_moist_coeff,
                diff_moist_coeff_max,
                diff_moist_damp_depth,
                nb,
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

        self._s_tnd = zeros(
            shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._su_tnd = zeros(
            shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._sv_tnd = zeros(
            shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        if self._moist:
            self._qv_tnd = zeros(
                shape, backend, dtype, default_origin, managed_memory=managed_memory
            )
            self._qc_tnd = zeros(
                shape, backend, dtype, default_origin, managed_memory=managed_memory
            )
            self._qr_tnd = zeros(
                shape, backend, dtype, default_origin, managed_memory=managed_memory
            )

    @property
    def input_properties(self):
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
    def tendency_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1 s^-1"}

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]

        self._core(in_s, self._s_tnd)
        self._core(in_su, self._su_tnd)
        self._core(in_sv, self._sv_tnd)

        return_dict = {
            "air_isentropic_density": self._s_tnd,
            "x_momentum_isentropic": self._su_tnd,
            "y_momentum_isentropic": self._sv_tnd,
        }

        if self._moist:
            in_qv = state[mfwv]
            in_qc = state[mfcw]
            in_qr = state[mfpw]

            self._core_moist(in_qv, self._qv_tnd)
            self._core_moist(in_qc, self._qc_tnd)
            self._core_moist(in_qr, self._qr_tnd)

            return_dict[mfwv] = self._qv_tnd
            return_dict[mfcw] = self._qc_tnd
            return_dict[mfpw] = self._qr_tnd

        return return_dict, {}
