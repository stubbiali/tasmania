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

import gridtools as gt
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.utils.storage_utils import get_storage_shape, zeros

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
        rebuild=False,
        storage_shape=None
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
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `numpy.dtype`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        halo : `tuple`, optional
            Storage halo.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple`, optional
            Shape of the storages.
        """
        self._moist = moist and smooth_moist_coeff is not None

        super().__init__(domain, "numerical")

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self.horizontal_boundary.nb

        shape = get_storage_shape(storage_shape, min_shape=(nx+1, ny+1, nz+1))

        self._core = HorizontalSmoothing.factory(
            smooth_type,
            shape,
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
                shape,
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

        self._out_s = zeros(shape, backend, dtype, halo=halo)
        self._out_su = zeros(shape, backend, dtype, halo=halo)
        self._out_sv = zeros(shape, backend, dtype, halo=halo)
        if self._moist:
            self._out_qv = zeros(shape, backend, dtype, halo=halo)
            self._out_qc = zeros(shape, backend, dtype, halo=halo)
            self._out_qr = zeros(shape, backend, dtype, halo=halo)

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
