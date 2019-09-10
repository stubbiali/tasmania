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
	IsentropicBoussinesqMinimalDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.isentropic.dynamics.boussinesq_minimal_prognostic import (
    IsentropicBoussinesqMinimalPrognostic,
)
from tasmania import get_dataarray_3d

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


# convenient shortcuts
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicBoussinesqMinimalDynamicalCore(DynamicalCore):
    """
	The three-dimensional (moist) isentropic, Boussinesq and minimal
	dynamical core. Here, *minimal* refers to the fact that only the
	horizontal advection is included in the so-called *dynamics*. Any
	other large-scale process (e.g., vertical advection, pressure gradient,
	Coriolis acceleration) might be included in the model only via parameterizations.
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
			See :class:`tasmania.IsentropicBoussinesqMinimalPrognostic`
			for all available options. Defaults to 'forward_euler'.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use. 
			See :class:`tasmania.HorizontalIsentropicBoussinesqMinimalFlux`
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
            "numerical",
            "s",
            intermediate_tendencies,
            intermediate_diagnostics,
            substeps,
            fast_tendencies,
            fast_diagnostics,
            dtype,
        )
        hb = self.horizontal_boundary

        #
        # prognostic
        #
        self._prognostic = IsentropicBoussinesqMinimalPrognostic.factory(
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

        #
        # vertical damping
        #
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        if damp:
            self._damper = VerticalDamping.factory(
                damp_type,
                (nx, ny, nz),
                self.grid,
                damp_depth,
                damp_max,
                time_units="s",
                backend=backend,
                dtype=dtype,
            )

        #
        # numerical smoothing
        #
        if smooth:
            self._smoother = HorizontalSmoothing.factory(
                smooth_type,
                (nx, ny, nz),
                smooth_coeff,
                smooth_coeff_max,
                smooth_damp_depth,
                hb.nb,
                backend,
                dtype,
            )
            if moist and smooth_moist:
                self._smoother_moist = HorizontalSmoothing.factory(
                    smooth_moist_type,
                    (nx, ny, nz),
                    smooth_moist_coeff,
                    smooth_moist_coeff_max,
                    smooth_moist_damp_depth,
                    hb.nb,
                    backend,
                    dtype,
                )

        #
        # diagnostics
        #
        self._velocity_components = HorizontalVelocity(
            self.grid, staggering=True, backend=backend, dtype=dtype
        )
        if moist:
            self._water_constituent = WaterConstituent(self.grid, backend, dtype)

        #
        # the method implementing each stage
        #
        self._array_call = self._array_call_dry if not moist else self._array_call_moist

        #
        # temporary and output arrays
        #
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        if damp:
            self._s_damped = np.zeros((nx, ny, nz), dtype=dtype)
            self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
            self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)
            self._ddmtg_damped = np.zeros((nx, ny, nz), dtype=dtype)
            if moist:
                self._qv_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._qc_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._qr_damped = np.zeros((nx, ny, nz), dtype=dtype)

        if smooth:
            self._s_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
            self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
            self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
            self._ddmtg_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

        if smooth_moist:
            self._qv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
            self._qc_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
            self._qr_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

        self._u_out = np.zeros((nx + 1, ny, nz), dtype=dtype)
        self._v_out = np.zeros((nx, ny + 1, nz), dtype=dtype)

        self._sqv_now = np.zeros((nx, ny, nz), dtype=dtype)
        self._sqc_now = np.zeros((nx, ny, nz), dtype=dtype)
        self._sqr_now = np.zeros((nx, ny, nz), dtype=dtype)

        self._qv_new = np.zeros((nx, ny, nz), dtype=dtype)
        self._qc_new = np.zeros((nx, ny, nz), dtype=dtype)
        self._qr_new = np.zeros((nx, ny, nz), dtype=dtype)

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
            "dd_montgomery_potential": {"dims": dims, "units": "m^2 K^-2 s^-2"},
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

    @property
    def _substep_input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        ftends, fdiags = self._fast_tends, self._fast_diags

        return_dict = {}

        if (
            ftends is not None
            and "air_isentropic_density" in ftends.input_properties
            or ftends is not None
            and "air_isentropic_density" in ftends.tendency_properties
            or fdiags is not None
            and "air_isentropic_density" in fdiags.input_properties
        ):
            return_dict["air_isentropic_density"] = {
                "dims": dims,
                "units": "kg m^-2 K^-1",
            }

        if (
            ftends is not None
            and "x_momentum_isentropic" in ftends.input_properties
            or ftends is not None
            and "x_momentum_isentropic" in ftends.tendency_properties
            or fdiags is not None
            and "x_momentum_isentropic" in fdiags.input_properties
        ):
            return_dict["x_momentum_isentropic"] = {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            }

        if (
            ftends is not None
            and "y_momentum_isentropic" in ftends.input_properties
            or ftends is not None
            and "y_momentum_isentropic" in ftends.tendency_properties
            or fdiags is not None
            and "y_momentum_isentropic" in fdiags.input_properties
        ):
            return_dict["y_momentum_isentropic"] = {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            }

        if self._moist:
            if (
                ftends is not None
                and mfwv in ftends.input_properties
                or ftends is not None
                and mfwv in ftends.tendency_properties
                or fdiags is not None
                and mfwv in fdiags.input_properties
            ):
                return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}

            if (
                ftends is not None
                and mfcw in ftends.input_properties
                or ftends is not None
                and mfcw in ftends.tendency_properties
                or fdiags is not None
                and mfcw in fdiags.input_properties
            ):
                return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}

            if (
                ftends is not None
                and mfpw in ftends.input_properties
                or ftends is not None
                and mfpw in ftends.tendency_properties
                or fdiags is not None
                and mfpw in fdiags.input_properties
            ):
                return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def _tendency_properties(self):
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
    def _substep_tendency_properties(self):
        return self._tendency_properties

    @property
    def _output_properties(self):
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
            "dd_montgomery_potential": {"dims": dims, "units": "m^2 K^-2 s^-2"},
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

    @property
    def _substep_output_properties(self):
        if not hasattr(self, "__substep_output_properties"):
            dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

            self.__substep_output_properties = {}

            if "air_isentropic_density" in self._substep_input_properties:
                self.__substep_output_properties["air_isentropic_density"] = {
                    "dims": dims,
                    "units": "kg m^-2 K^-1",
                }

            if "x_momentum_isentropic" in self._substep_input_properties:
                self.__substep_output_properties["x_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                }

            if "y_momentum_isentropic" in self._substep_input_properties:
                self.__substep_output_properties["y_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                }

            if self._moist:
                if mfwv in self._substep_input_properties:
                    self.__substep_output_properties[mfwv] = {
                        "dims": dims,
                        "units": "g g^-1",
                    }

                if mfcw in self._substep_input_properties:
                    self.__substep_output_properties[mfcw] = {
                        "dims": dims,
                        "units": "g g^-1",
                    }

                if mfpw in self._substep_input_properties:
                    self.__substep_output_properties[mfpw] = {
                        "dims": dims,
                        "units": "g g^-1",
                    }

        return self.__substep_output_properties

    @property
    def stages(self):
        return self._prognostic.stages

    @property
    def substep_fractions(self):
        return self._prognostic.substep_fractions

    def _allocate_output_state(self):
        """
		Allocate memory only for the prognostic fields.
		"""
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        dtype = self._dtype

        out_state = {}

        names = [
            "air_isentropic_density",
            "dd_montgomery_potential",
            "x_velocity_at_u_locations",
            "x_momentum_isentropic",
            "y_velocity_at_v_locations",
            "y_momentum_isentropic",
        ]
        if self._moist:
            names.append(mfwv)
            names.append(mfcw)
            names.append(mfpw)

        for name in names:
            dims = self.output_properties[name]["dims"]
            units = self.output_properties[name]["units"]

            shape = (
                nx + 1 if "at_u_locations" in dims[0] else nx,
                ny + 1 if "at_v_locations" in dims[1] else ny,
                nz + 1 if "on_interface_levels" in dims[2] else nz,
            )

            out_state[name] = get_dataarray_3d(
                np.zeros(shape, dtype=dtype), g, units, name=name
            )

        return out_state

    def array_call(self, stage, raw_state, raw_tendencies, timestep):
        return self._array_call(stage, raw_state, raw_tendencies, timestep)

    def _array_call_dry(self, stage, raw_state, raw_tendencies, timestep):
        """
		Perform a stage of the dry dynamical core.
		"""
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref = (
                    ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
                )
                self._su_ref = (
                    ref_state["x_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._sv_ref = (
                    ref_state["y_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._ddmtg_ref = (
                    ref_state["dd_montgomery_potential"].to_units("m^2 K^-2 s^-2").values
                )
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the horizontal "
                    "boundary conditions, but needed by the wave absorber."
                )

            # save the current solution
            self._s_now = raw_state["air_isentropic_density"]
            self._su_now = raw_state["x_momentum_isentropic"]
            self._sv_now = raw_state["y_momentum_isentropic"]
            self._ddmtg_now = raw_state["dd_montgomery_potential"]

        # perform the prognostic step
        raw_state_new = self._prognostic.stage_call(
            stage, timestep, raw_state, raw_tendencies
        )

        # apply the lateral boundary conditions
        hb.dmn_enforce_raw(raw_state_new, out_properties)

        damped = False
        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            damped = True

            # extract the stepped prognostic model variables
            s_new = raw_state_new["air_isentropic_density"]
            su_new = raw_state_new["x_momentum_isentropic"]
            sv_new = raw_state_new["y_momentum_isentropic"]
            ddmtg_new = raw_state_new["dd_montgomery_potential"]

            # apply vertical damping
            self._damper(timestep, self._s_now, s_new, self._s_ref, self._s_damped)
            self._damper(timestep, self._su_now, su_new, self._su_ref, self._su_damped)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, self._sv_damped)
            self._damper(
                timestep, self._ddmtg_now, ddmtg_new, self._ddmtg_ref, self._ddmtg_damped
            )

        # properly set pointers to current solution
        s_new = self._s_damped if damped else raw_state_new["air_isentropic_density"]
        su_new = self._su_damped if damped else raw_state_new["x_momentum_isentropic"]
        sv_new = self._sv_damped if damped else raw_state_new["y_momentum_isentropic"]
        ddmtg_new = (
            self._ddmtg_damped if damped else raw_state_new["dd_montgomery_potential"]
        )

        smoothed = False
        if self._smooth and (self._smooth_at_every_stage or stage == self.stages - 1):
            smoothed = True

            # apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)
            self._smoother(ddmtg_new, self._ddmtg_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                "air_isentropic_density": self._s_smoothed,
                "x_momentum_isentropic": self._su_smoothed,
                "y_momentum_isentropic": self._sv_smoothed,
                "dd_montgomery_potential": self._ddmtg_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new
        ddmtg_out = self._ddmtg_smoothed if smoothed else ddmtg_new

        # diagnose the velocity components
        self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out, self._u_out, self._v_out
        )
        hb.dmn_set_outermost_layers_x(
            self._u_out,
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=raw_state_new["time"],
        )
        hb.dmn_set_outermost_layers_y(
            self._v_out,
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=raw_state_new["time"],
        )

        # instantiate the output state
        raw_state_out = {
            "time": raw_state_new["time"],
            "air_isentropic_density": s_out,
            "dd_montgomery_potential": ddmtg_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": self._u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": self._v_out,
        }

        return raw_state_out

    def _array_call_moist(self, stage, raw_state, raw_tendencies, timestep):
        """
		Perform a stage of the moist dynamical core.
		"""
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref = (
                    ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
                )
                self._su_ref = (
                    ref_state["x_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._sv_ref = (
                    ref_state["y_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._ddmtg_ref = (
                    ref_state["dd_montgomery_potential"].to_units("m^2 K^-2 s^-2").values
                )
                self._qv_ref = ref_state[mfwv].to_units("g g^-1").values
                self._qc_ref = ref_state[mfcw].to_units("g g^-1").values
                self._qr_ref = ref_state[mfpw].to_units("g g^-1").values
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the horizontal "
                    "boundary conditions, but needed by the wave absorber."
                )

            # save the current solution
            self._s_now = raw_state["air_isentropic_density"]
            self._su_now = raw_state["x_momentum_isentropic"]
            self._sv_now = raw_state["y_momentum_isentropic"]
            self._ddmtg_now = raw_state["dd_montgomery_potential"]
            self._qv_now = raw_state[mfwv]
            self._qc_now = raw_state[mfcw]
            self._qr_now = raw_state[mfpw]

        # diagnose the isentropic density of all water constituents
        s_now = raw_state["air_isentropic_density"]
        qv_now = raw_state[mfwv]
        qc_now = raw_state[mfcw]
        qr_now = raw_state[mfpw]
        self._water_constituent.get_density_of_water_constituent(
            s_now, qv_now, self._sqv_now
        )
        self._water_constituent.get_density_of_water_constituent(
            s_now, qc_now, self._sqc_now
        )
        self._water_constituent.get_density_of_water_constituent(
            s_now, qr_now, self._sqr_now
        )
        raw_state["isentropic_density_of_water_vapor"] = self._sqv_now
        raw_state["isentropic_density_of_cloud_liquid_water"] = self._sqc_now
        raw_state["isentropic_density_of_precipitation_water"] = self._sqr_now

        # perform the prognostic step
        raw_state_new = self._prognostic.stage_call(
            stage, timestep, raw_state, raw_tendencies
        )

        # diagnose the mass fraction of all water constituents
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_water_vapor"],
            self._qv_new,
            clipping=True,
        )
        raw_state_new[mfwv] = self._qv_new
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_cloud_liquid_water"],
            self._qc_new,
            clipping=True,
        )
        raw_state_new[mfcw] = self._qc_new
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_precipitation_water"],
            self._qr_new,
            clipping=True,
        )
        raw_state_new[mfpw] = self._qr_new

        # apply the lateral boundary conditions
        hb.dmn_enforce_raw(raw_state_new, out_properties)

        damped = False
        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            damped = True

            # extract the stepped prognostic model variables
            s_new = raw_state_new["air_isentropic_density"]
            su_new = raw_state_new["x_momentum_isentropic"]
            sv_new = raw_state_new["y_momentum_isentropic"]
            ddmtg_new = raw_state_new["dd_montgomery_potential"]

            # apply vertical damping
            self._damper(timestep, self._s_now, s_new, self._s_ref, self._s_damped)
            self._damper(timestep, self._su_now, su_new, self._su_ref, self._su_damped)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, self._sv_damped)
            self._damper(
                timestep, self._ddmtg_now, ddmtg_new, self._ddmtg_ref, self._ddmtg_damped
            )

        # properly set pointers to current solution
        s_new = self._s_damped if damped else raw_state_new["air_isentropic_density"]
        su_new = self._su_damped if damped else raw_state_new["x_momentum_isentropic"]
        sv_new = self._sv_damped if damped else raw_state_new["y_momentum_isentropic"]
        ddmtg_new = (
            self._ddmtg_damped if damped else raw_state_new["dd_montgomery_potential"]
        )

        smoothed = False
        if self._smooth and (self._smooth_at_every_stage or stage == self.stages - 1):
            smoothed = True

            # apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)
            self._smoother(ddmtg_new, self._ddmtg_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                "air_isentropic_density": self._s_smoothed,
                "x_momentum_isentropic": self._su_smoothed,
                "y_momentum_isentropic": self._sv_smoothed,
                "dd_montgomery_potential": self._ddmtg_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new
        ddmtg_out = self._ddmtg_smoothed if smoothed else ddmtg_new

        smoothed_moist = False
        if self._smooth_moist and (
            self._smooth_moist_at_every_stage or stage == self.stages - 1
        ):
            smoothed_moist = True

            # apply horizontal smoothing
            self._smoother(self._qv_new, self._qv_smoothed)
            self._smoother(self._qc_new, self._qc_smoothed)
            self._smoother(self._qr_new, self._qr_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                mfwv: self._qv_smoothed,
                mfcw: self._qc_smoothed,
                mfpw: self._qr_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        qv_out = self._qv_smoothed if smoothed_moist else self._qv_new
        qc_out = self._qc_smoothed if smoothed_moist else self._qc_new
        qr_out = self._qr_smoothed if smoothed_moist else self._qr_new

        # diagnose the velocity components
        self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out, self._u_out, self._v_out
        )
        hb.dmn_set_outermost_layers_x(
            self._u_out,
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=raw_state_new["time"],
        )
        hb.dmn_set_outermost_layers_y(
            self._v_out,
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=raw_state_new["time"],
        )

        # instantiate the output state
        raw_state_out = {
            "time": raw_state_new["time"],
            "air_isentropic_density": s_out,
            "dd_montgomery_potential": ddmtg_out,
            mfwv: qv_out,
            mfcw: qc_out,
            mfpw: qr_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": self._u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": self._v_out,
        }

        return raw_state_out

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
