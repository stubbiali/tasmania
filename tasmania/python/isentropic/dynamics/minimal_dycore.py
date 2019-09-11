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
	IsentropicMinimalDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore
from tasmania.python.isentropic.dynamics.minimal_prognostic import (
    IsentropicMinimalPrognostic,
)

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


# convenient shortcuts
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicMinimalDynamicalCore(IsentropicDynamicalCore):
    """
	The three-dimensional (moist) isentropic minimal dynamical core.
	Here, *minimal* refers to the fact that only the horizontal advection
	is included in the so-called *dynamics*. Any other large-scale process
	(e.g., vertical advection, pressure gradient, Coriolis acceleration) might
	be included in the model only via parameterizations.
	The conservative form of the governing equations is used.
	"""

    def __init__(
        self,
        domain,
        intermediate_tendencies=None,
        intermediate_diagnostics=None,
        substeps=0,
        fast_tendencies=None,
        fast_diagnostics=None,
        moist=False,
        time_integration_scheme="forward_euler",
        horizontal_flux_scheme="upwind",
        damp=True,
        damp_at_every_stage=True,
        damp_type="rayleigh",
        damp_depth=15,
        damp_max=0.0002,
        smooth=True,
        smooth_at_every_stage=True,
        smooth_type="first_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=10,
        smooth_moist=False,
        smooth_moist_at_every_stage=True,
        smooth_moist_type="first_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=10,
        backend=gt.mode.NUMPY,
        dtype=datatype,
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		intermediate_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the intermediate physical tendencies.
			Here, *intermediate* refers to the fact that these physical
			packages are called *before* each stage of the dynamical core
			to calculate the physical tendencies.
		intermediate_diagnostics : `obj`, optional
			An instance of either

				* :class:`sympl.DiagnosticComponent`,
				* :class:`sympl.DiagnosticComponentComposite`, or
				* :class:`tasmania.DiagnosticComponentComposite`

			retrieving diagnostics at the end of each stage, once the
			sub-timestepping routine is over.
		substeps : `int`, optional
			Number of sub-steps to perform. Defaults to 0, meaning that no
			sub-stepping technique is implemented.
		fast_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the fast physical tendencies.
			Here, *fast* refers to the fact that these physical packages are
			called *before* each sub-step of any stage of the dynamical core
			to calculate the physical tendencies.
			This parameter is ignored if `substeps` argument is not positive.
		fast_diagnostics : `obj`, optional
			An instance of either

				* :class:`sympl.DiagnosticComponent`,
				* :class:`sympl.DiagnosticComponentComposite`, or
				* :class:`tasmania.DiagnosticComponentComposite`

			retrieving diagnostics at the end of each sub-step of any stage
			of the dynamical core.
			This parameter is ignored if `substeps` argument is not positive.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		time_integration_scheme : str
			String specifying the time stepping method to implement. 
			See :class:`tasmania.IsentropicMinimalPrognostic`
			for all available options. Defaults to 'forward_euler'.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use. 
			See :class:`tasmania.HorizontalIsentropicMinimalFlux`
			for all available options. Defaults to 'upwind'.
		damp : `bool`, optional
			:obj:`True` to enable vertical damping, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		damp_at_every_stage : `bool`, optional
			:obj:`True` to carry out the damping at each stage of the multi-stage
			time-integrator, :obj:`False` to carry out the damping only at the end
			of each timestep. Defaults to :obj:`True`.
		damp_type : `str`, optional
			String specifying the vertical damping scheme to implement.
			See :class:`tasmania.VerticalDamping` for all available options.
			Defaults to 'rayleigh'.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		smooth : `bool`, optional
			:obj:`True` to enable horizontal numerical smoothing, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		smooth_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing at each stage of the time-
			integrator, :obj:`False` to apply numerical smoothing only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first_order'.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Defaults to 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. 
			See :class:`tasmania.HorizontalSmoothing` for further details.
			Defaults to 0.24.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Defaults to 10.
		smooth_moist : `bool`, optional
			:obj:`True` to enable horizontal numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_moist_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing on the water constituents
			at each stage of the time-integrator, :obj:`False` to apply numerical
			smoothing only at the end of each timestep. Defaults to :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply on the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first-order'. 
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for further details. 
			Defaults to 0.24. 
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		"""
        #
        # input parameters
        #
        self._moist = moist
        self._damp = damp
        self._damp_at_every_stage = damp_at_every_stage
        self._smooth = smooth
        self._smooth_at_every_stage = smooth_at_every_stage
        self._smooth_moist = smooth_moist
        self._smooth_moist_at_every_stage = smooth_moist_at_every_stage
        self._dtype = dtype

        #
        # parent constructor
        #
        super().__init__(
            domain,
            intermediate_tendencies,
            intermediate_diagnostics,
            substeps,
            fast_tendencies,
            fast_diagnostics,
            moist=moist,
            damp=damp,
            damp_at_every_stage=damp_at_every_stage,
            damp_type=damp_type,
            damp_depth=damp_depth,
            damp_max=damp_max,
            smooth=smooth,
            smooth_at_every_stage=smooth_at_every_stage,
            smooth_type=smooth_type,
            smooth_coeff=smooth_coeff,
            smooth_coeff_max=smooth_coeff_max,
            smooth_damp_depth=smooth_damp_depth,
            smooth_moist=smooth_moist,
            smooth_moist_at_every_stage=smooth_moist_at_every_stage,
            smooth_moist_type=smooth_moist_type,
            smooth_moist_coeff=smooth_moist_coeff,
            smooth_moist_coeff_max=smooth_moist_coeff_max,
            smooth_moist_damp_depth=smooth_moist_damp_depth,
            backend=backend,
            dtype=dtype,
        )

        #
        # prognostic
        #
        self._prognostic = IsentropicMinimalPrognostic.factory(
            time_integration_scheme,
            horizontal_flux_scheme,
            "xy",
            self.grid,
            self.horizontal_boundary,
            moist,
            substeps,
            backend,
            dtype,
        )
        self._prognostic.substep_output_properties = self._substep_output_properties

    @property
    def _input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        dims_stg_x = (
            self.grid.x_at_u_locations.dims[0],
            self.grid.y.dims[0],
            self.grid.z.dims[0],
        )
        dims_stg_y = (
            self.grid.x.dims[0],
            self.grid.y_at_v_locations.dims[0],
            self.grid.z.dims[0],
        )

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "x_velocity_at_u_locations": {"dims": dims_stg_x, "units": "m s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_velocity_at_v_locations": {"dims": dims_stg_y, "units": "m s^-1"},
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    def substep_array_call(
        self,
        stage,
        substep,
        raw_state,
        raw_stage_state,
        raw_tmp_state,
        raw_tendencies,
        timestep,
    ):
        # merely perform the sub-step
        return self._prognostic.substep_call(
            stage,
            substep,
            timestep,
            raw_state,
            raw_stage_state,
            raw_tmp_state,
            raw_tendencies,
        )
