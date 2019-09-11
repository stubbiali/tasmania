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
    IsentropicHorizontalSmoothing(DiagnosticComponent)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float64


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
        domain,
        smooth_type,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        moist=False,
        smooth_moist_coeff=None,
        smooth_moist_coeff_max=None,
        smooth_moist_damp_depth=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        halo=None,
        rebuild=False
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
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
            :obj:`True` if water species are included in the model and should
            be smoothed, :obj:`False` otherwise. Defaults to :obj:`False`.
        smooth_moist_coeff : `float`, optional
            The smoothing coefficient for the water species.
        smooth_moist_coeff_max : `float`, optional
            The maximum value assumed by the smoothing coefficient for the water
            species close to the upper boundary.
        smooth_damp_depth : int
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
        """
        self._moist = moist and smooth_moist_coeff is not None

        super().__init__(domain, "numerical")

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self.horizontal_boundary.nb

        self._core = HorizontalSmoothing.factory(
            smooth_type,
            (nx, ny, nz),
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
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
                (nx, ny, nz),
                smooth_moist_coeff,
                smooth_moist_coeff_max,
                smooth_moist_damp_depth,
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
        self._out_s = gt.storage.zeros(descriptor, backend=backend)
        self._in_su = gt.storage.zeros(descriptor, backend=backend)
        self._out_su = gt.storage.zeros(descriptor, backend=backend)
        self._in_sv = gt.storage.zeros(descriptor, backend=backend)
        self._out_sv = gt.storage.zeros(descriptor, backend=backend)
        if self._moist:
            self._in_qv = gt.storage.zeros(descriptor, backend=backend)
            self._out_qv = gt.storage.zeros(descriptor, backend=backend)
            self._in_qc = gt.storage.zeros(descriptor, backend=backend)
            self._out_qc = gt.storage.zeros(descriptor, backend=backend)
            self._in_qr = gt.storage.zeros(descriptor, backend=backend)
            self._out_qr = gt.storage.zeros(descriptor, backend=backend)

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
    def diagnostic_properties(self):
        return self.input_properties

    def array_call(self, state):
        self._in_s.data[...] = state["air_isentropic_density"]
        self._in_su.data[...] = state["x_momentum_isentropic"]
        self._in_sv.data[...] = state["y_momentum_isentropic"]
        if self._moist:
            self._in_qv.data[...] = state[mfwv]
            self._in_qc.data[...] = state[mfcw]
            self._in_qr.data[...] = state[mfpw]

        self._core(self._in_s, self._out_s)
        self._core(self._in_su, self._out_su)
        self._core(self._in_sv, self._out_sv)

        return_dict = {
            "air_isentropic_density": self._out_s.data,
            "x_momentum_isentropic": self._out_su.data,
            "y_momentum_isentropic": self._out_sv.data,
        }

        if self._moist:
            self._core_moist(self._in_qv, self._out_qv)
            self._core_moist(self._in_qc, self._out_qc)
            self._core_moist(self._in_qr, self._out_qr)

            return_dict[mfwv] = self._out_qv.data
            return_dict[mfcw] = self._out_qc.data
            return_dict[mfpw] = self._out_qr.data

        return return_dict
