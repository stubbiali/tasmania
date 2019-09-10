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
	IsentropicHorizontalDiffusion(TendencyComponent)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils.storage_utils import get_storage_descriptor

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
        halo=None,
        rebuild=False,
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
			:obj:`True` if water species are included in the model and should
			be diffused, :obj:`False` otherwise. Defaults to :obj:`False`.
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
			Keyword arguments to be directly forwarded to the parent constructor.
		"""
        self._moist = moist and diffusion_moist_coeff is not None

        super().__init__(domain, "numerical", **kwargs)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        nb = self.horizontal_boundary.nb

        diff_coeff = diffusion_coeff.to_units("s^-1").values.item()
        diff_coeff_max = diffusion_coeff_max.to_units("s^-1").values.item()

        self._core = HorizontalDiffusion.factory(
            diffusion_type,
            (nx, ny, nz),
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
            halo=halo,
            rebuild=rebuild,
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
                (nx, ny, nz),
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
                halo=halo,
                rebuild=rebuild,
            )
        else:
            self._core_moist = None

        descriptor = get_storage_descriptor((nx, ny, nz), dtype, halo=halo)
        self._in_s = gt.storage.zeros(descriptor, backend=backend)
        self._s_tnd = gt.storage.zeros(descriptor, backend=backend)
        self._in_su = gt.storage.zeros(descriptor, backend=backend)
        self._su_tnd = gt.storage.zeros(descriptor, backend=backend)
        self._in_sv = gt.storage.zeros(descriptor, backend=backend)
        self._sv_tnd = gt.storage.zeros(descriptor, backend=backend)
        if self._moist:
            self._in_qv = gt.storage.zeros(descriptor, backend=backend)
            self._qv_tnd = gt.storage.zeros(descriptor, backend=backend)
            self._in_qc = gt.storage.zeros(descriptor, backend=backend)
            self._qc_tnd = gt.storage.zeros(descriptor, backend=backend)
            self._in_qr = gt.storage.zeros(descriptor, backend=backend)
            self._qr_tnd = gt.storage.zeros(descriptor, backend=backend)

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
        self._in_s.data[...] = state["air_isentropic_density"]
        self._in_su.data[...] = state["x_momentum_isentropic"]
        self._in_sv.data[...] = state["y_momentum_isentropic"]
        if self._moist:
            self._in_qv.data[...] = state[mfwv]
            self._in_qc.data[...] = state[mfcw]
            self._in_qr.data[...] = state[mfpw]

        self._core(self._in_s, self._s_tnd)
        self._core(self._in_su, self._su_tnd)
        self._core(self._in_sv, self._sv_tnd)

        return_dict = {
            "air_isentropic_density": self._s_tnd.data,
            "x_momentum_isentropic": self._su_tnd.data,
            "y_momentum_isentropic": self._sv_tnd.data,
        }

        if self._moist:
            self._core_moist(self._in_qv, self._qv_tnd)
            self._core_moist(self._in_qc, self._qc_tnd)
            self._core_moist(self._in_qr, self._qr_tnd)

            return_dict[mfwv] = self._qv_tnd.data
            return_dict[mfcw] = self._qc_tnd.data
            return_dict[mfpw] = self._qr_tnd.data

        return return_dict, {}
