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
	IsentropicBoussinesqMinimalPrognostic
"""
import abc
import numpy as np

import gridtools as gt
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicBoussinesqMinimalHorizontalFlux,
)

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicBoussinesqMinimalPrognostic:
    """
	Abstract base class whose derived classes implement different
	schemes to carry out the prognostic steps of the three-dimensional
	isentropic, Boussinesq and minimal dynamical core. Here, *minimal* means
	that only horizontal advection is integrated within the dynamical core.
	The vertical advection, the Coriolis acceleration, the pressure gradient
	and the sedimentation motion are not included in the dynamics, but
	rather parameterized. The conservative form of the governing equations is used.
	"""

    # make the class abstract
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        horizontal_flux_scheme,
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend,
        dtype=datatype,
    ):
        """
		Parameters
		----------
		horizontal_flux_scheme : str
			The numerical horizontal flux scheme to implement.
			See :class:`~tasmania.IsentropicBoussinesqMinimalHorizontalFlux`
			for the complete list of the available options.
		mode : str
			Either

				* 'x', to integrate only the x-advection,
				* 'y', to integrate only the y-advection, or
				* 'xy', to integrate both the x- and the y-advection.

		grid : tasmania.Grid
			The underlying grid.
		hb : tasmania.HorizontalBoundary
			The object handling the lateral boundary conditions.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		substeps : int
			The number of substeps to perform.
		backend : obj
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
        # keep track of the input parameters
        self._hflux_scheme = horizontal_flux_scheme
        self._mode = mode if mode in ["x", "y", "xy"] else "xy"
        self._grid = grid
        self._hb = hb
        self._moist = moist
        self._substeps = substeps
        self._backend = backend
        self._dtype = dtype

        # instantiate the class computing the numerical horizontal fluxes
        self._hflux = IsentropicBoussinesqMinimalHorizontalFlux.factory(
            self._hflux_scheme, grid, moist
        )
        assert hb.nb >= self._hflux.extent, (
            "The number of lateral boundary layers is {}, but should be "
            "greater or equal than {}.".format(hb.nb, self._hflux.extent)
        )
        if mode != "y":
            assert grid.nx >= 2 * hb.nb + 1, (
                "The number of grid points along the first horizontal "
                "dimension is {}, but should be greater or equal than {}.".format(
                    grid.nx, 2 * hb.nb + 1
                )
            )
        if mode != "x":
            assert grid.ny >= 2 * hb.nb + 1, (
                "The number of grid points along the second horizontal "
                "dimension is {}, but should be greater or equal than {}.".format(
                    grid.ny, 2 * hb.nb + 1
                )
            )

        # initialize properties dictionary
        self._substep_output_properties = None

        # initialize the pointer to the underlying GT4Py stencil in charge
        # of carrying out the sub-steps
        if substeps > 0:
            self._substep_stencil = None

    @property
    @abc.abstractmethod
    def stages(self):
        """
		Return
		------
		int :
			The number of stages performed by the time-integration scheme.
		"""

    @property
    @abc.abstractmethod
    def substep_fractions(self):
        """
		Return
		------
		float or tuple :
			In a partial time splitting framework, for each stage, fraction of the
			total number of substeps to carry out.
		"""

    @property
    def substep_output_properties(self):
        """
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any substep, and whose
			values are fundamental properties (dims, units) of those variables.
		"""
        if self._substep_output_properties is None:
            raise RuntimeError("substep_output_properties required but not set.")
        return self._substep_output_properties

    @substep_output_properties.setter
    def substep_output_properties(self, value):
        """
		Parameters
		----------
		value : dict
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any substep, and whose
			values are fundamental properties (dims, units) of those variables.
		"""
        self._substep_output_properties = value

    @abc.abstractmethod
    def stage_call(self, stage, timestep, state, tendencies=None):
        """
		Perform a stage.

		Parameters
		----------
		stage : int
			The stage to perform.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
		state : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing the values
			for those variables.
		tendencies : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing (slow and
			intermediate) physical tendencies for those variables.

		Return
		------
		dict :
			Dictionary whose keys are strings indicating the conservative
			prognostic model variables, and values are :class:`numpy.ndarray`\s
			containing new values for those variables.
		"""
        pass

    def substep_call(
        self, stage, substep, timestep, state, stage_state, tmp_state, tendencies=None
    ):
        """
		Perform a sub-step.

		Parameters
		----------
		stage : int
			The stage to perform.
		substep : int
			The substep to perform.
		timestep : datetime.timedelta
			The time step.
		state : dict
			The raw state at the current *main* time level.
		stage_state : dict
			The (raw) state dictionary returned by the latest stage.
		tmp_state : dict
			The raw state to sub-step.
		tendencies : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing
			(fast) tendencies for those values.

		Return
		------
		dict :
			Dictionary whose keys are strings indicating the conservative
			prognostic model variables, and values are :class:`numpy.ndarray`\s
			containing new values for those variables.
		"""
        # initialize the stencil object
        if self._substep_stencil is None:
            self._substep_stencil_initialize(tendencies)

        # set the stencil's inputs
        self._substep_stencil_set_inputs(
            stage, substep, timestep, state, stage_state, tmp_state, tendencies
        )

        # run the stencil
        self._substep_stencil.compute()

        # compose the output state
        out_time = (
            state["time"] + timestep / self._substeps
            if substep == 0
            else tmp_state["time"] + timestep / self._substeps
        )
        out_state = {"time": out_time}
        for key in self._substep_output_properties:
            out_state[key] = self._substep_stencil_outputs[key]

        return out_state

    @staticmethod
    def factory(
        time_integration_scheme,
        horizontal_flux_scheme,
        mode,
        grid,
        hb,
        moist=False,
        substeps=0,
        backend=gt.mode.NUMPY,
        dtype=datatype,
    ):
        """
		Static method returning an instance of the derived class implementing
		the time stepping scheme specified by ``time_scheme``.

		Parameters
		----------
		time_integration_scheme : str
			The time stepping method to implement. Available options are:

				* 'forward_euler', for the forward Euler scheme;
				* 'centered', for a centered scheme;
				* 'rk2', for the two-stages, second-order Runge-Kutta (RK) scheme;
				* 'rk3ws', for the three-stages RK scheme as used in the
					`COSMO model <http://www.cosmo-model.org>`_; this method is
					nominally second-order, and third-order for linear problems;
				* 'rk3', for the three-stages, third-order RK scheme.

		horizontal_flux_scheme : str
			The numerical horizontal flux scheme to implement.
			See :class:`~tasmania.IsentropicBoussinesqMinimalHorizontalFlux`
			for the complete list of the available options.
		mode : str
			Either

				* 'x', to integrate only the x-advection,
				* 'y', to integrate only the y-advection, or
				* 'xy', to integrate both the x- and the y-advection.

		grid : tasmania.Grid
			The underlying grid.
		hb : tasmania.HorizontalBoundary
			The object handling the lateral boundary conditions.
		moist : `bool`, optional
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		substeps : int
			The number of substeps to perform.
		backend : obj
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.

		Return
		------
		obj :
			An instance of the derived class implementing ``time_integration_scheme``.
		"""
        from .implementations.boussinesq_minimal_prognostic import (
            ForwardEuler,
            Centered,
            RK2,
            RK3WS,
            RK3,
        )

        args = (horizontal_flux_scheme, mode, grid, hb, moist, substeps, backend, dtype)

        if time_integration_scheme == "forward_euler":
            return ForwardEuler(*args)
        elif time_integration_scheme == "centered":
            return Centered(*args)
        elif time_integration_scheme == "rk2":
            return RK2(*args)
        elif time_integration_scheme == "rk3ws":
            return RK3WS(*args)
        elif time_integration_scheme == "rk3":
            return RK3(*args)
        else:
            raise ValueError(
                "Unknown time integration scheme {}. Available options are: "
                "forward_euler, centered, rk2, rk3ws, rk3.".format(
                    time_integration_scheme
                )
            )

    def _stage_stencil_allocate_inputs(self, tendencies):
        """
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which implement the stages.
		"""
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dtype = self._dtype
        tendency_names = () if tendencies is None else tendencies.keys()

        # instantiate a GT4Py Global representing the timestep
        self._dt = gt.Global()

        # allocate the Numpy arrays which will store the current solution
        # and serve as stencil's inputs
        self._in_s = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_u = np.zeros((nx + 1, ny, nz), dtype=dtype)
        self._in_v = np.zeros((nx, ny + 1, nz), dtype=dtype)
        self._in_su = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_sv = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_ddmtg = np.zeros((nx, ny, nz), dtype=dtype)
        if self._moist:
            self._in_sqv = np.zeros((nx, ny, nz), dtype=dtype)
            self._in_sqc = np.zeros((nx, ny, nz), dtype=dtype)
            self._in_sqr = np.zeros((nx, ny, nz), dtype=dtype)

        # allocate the input Numpy arrays which will store the tendencies
        # and serve as stencil's inputs
        if tendency_names is not None:
            if "air_isentropic_density" in tendency_names:
                self._in_s_tnd = np.zeros((nx, ny, nz), dtype=dtype)
            if "x_momentum_isentropic" in tendency_names:
                self._in_su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
            if "y_momentum_isentropic" in tendency_names:
                self._in_sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
            if mfwv in tendency_names:
                self._in_qv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
            if mfcw in tendency_names:
                self._in_qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
            if mfpw in tendency_names:
                self._in_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)

    def _stage_stencil_allocate_outputs(self):
        """
		Allocate the Numpy arrays which serve as outputs for the GT4Py stencils
		which perform the stages.
		"""
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dtype = self._dtype

        # allocate the Numpy arrays which will serve as stencil's outputs
        self._out_s = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_ddmtg = np.zeros((nx, ny, nz), dtype=dtype)
        if self._moist:
            self._out_sqv = np.zeros((nx, ny, nz), dtype=dtype)
            self._out_sqc = np.zeros((nx, ny, nz), dtype=dtype)
            self._out_sqr = np.zeros((nx, ny, nz), dtype=dtype)

    def _stage_stencil_set_inputs(self, stage, timestep, state, tendencies):
        """
		Update the attributes which serve as inputs to the GT4Py stencils
		which perform the stages.
		"""
        # shortcuts
        if tendencies is not None:
            s_tnd_on = tendencies.get("air_isentropic_density", None) is not None
            su_tnd_on = tendencies.get("x_momentum_isentropic", None) is not None
            sv_tnd_on = tendencies.get("y_momentum_isentropic", None) is not None
            qv_tnd_on = tendencies.get(mfwv, None) is not None
            qc_tnd_on = tendencies.get(mfcw, None) is not None
            qr_tnd_on = tendencies.get(mfpw, None) is not None
        else:
            s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

        # update the local time step
        self._dt.value = timestep.total_seconds()

        # update the Numpy arrays which serve as inputs to the GT4Py stencils
        self._in_s[...] = state["air_isentropic_density"][...]
        self._in_u[...] = state["x_velocity_at_u_locations"][...]
        self._in_v[...] = state["y_velocity_at_v_locations"][...]
        self._in_su[...] = state["x_momentum_isentropic"][...]
        self._in_sv[...] = state["y_momentum_isentropic"][...]
        self._in_ddmtg[...] = state["dd_montgomery_potential"][...]
        if self._moist:
            self._in_sqv[...] = state["isentropic_density_of_water_vapor"][...]
            self._in_sqc[...] = state["isentropic_density_of_cloud_liquid_water"][...]
            self._in_sqr[...] = state["isentropic_density_of_precipitation_water"][...]
        if s_tnd_on:
            self._in_s_tnd[...] = tendencies["air_isentropic_density"][...]
        if su_tnd_on:
            self._in_su_tnd[...] = tendencies["x_momentum_isentropic"][...]
        if sv_tnd_on:
            self._in_sv_tnd[...] = tendencies["y_momentum_isentropic"][...]
        if qv_tnd_on:
            self._in_qv_tnd[...] = tendencies[mfwv][...]
        if qc_tnd_on:
            self._in_qc_tnd[...] = tendencies[mfcw][...]
        if qr_tnd_on:
            self._in_qr_tnd[...] = tendencies[mfpw][...]

    def _substep_stencil_initialize(self, tendencies):
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dtype = self._dtype
        tendency_names = () if tendencies is None else tendencies.keys()

        self._dts = gt.Global()
        self._stage_substeps = gt.Global()

        self._substep_stencil_state_inputs = {}
        self._substep_stencil_stage_state_inputs = {}
        self._substep_stencil_tmp_state_inputs = {}
        self._substep_stencil_tendencies_inputs = {}
        self._substep_stencil_outputs = {}

        inputs = {}
        outputs = {}

        shorthands = {
            "air_isentropic_density": "s",
            "x_momentum_isentropic": "su",
            "y_momentum_isentropic": "sv",
            "mass_fraction_of_water_vapor_in_air": "qv",
            "mass_fraction_of_cloud_liquid_water_in_air": "qc",
            "mass_fraction_of_precipitation_water_in_air": "qr",
        }

        for var, shand in shorthands.items():
            if var in self._substep_output_properties:
                self._substep_stencil_state_inputs[var] = np.zeros(
                    (nx, ny, nz), dtype=dtype
                )
                inputs[shand] = self._substep_stencil_state_inputs[var]

                self._substep_stencil_stage_state_inputs[var] = np.zeros(
                    (nx, ny, nz), dtype=dtype
                )
                inputs["stage_" + shand] = self._substep_stencil_stage_state_inputs[var]

                self._substep_stencil_tmp_state_inputs[var] = np.zeros(
                    (nx, ny, nz), dtype=dtype
                )
                inputs["tmp_" + shand] = self._substep_stencil_tmp_state_inputs[var]

                if var in tendency_names:
                    self._substep_stencil_tendencies_inputs[var] = np.zeros(
                        (nx, ny, nz), dtype=dtype
                    )
                    inputs["tnd_" + shand] = self._substep_stencil_tendencies_inputs[
                        var
                    ]

                self._substep_stencil_outputs[var] = np.zeros((nx, ny, nz), dtype=dtype)
                outputs["out_" + shand] = self._substep_stencil_outputs[var]

        self._substep_stencil = gt.NGStencil(
            definitions_func=self.__class__._substep_stencil_defs,
            inputs=inputs,
            global_inputs={"dts": self._dts, "substeps": self._stage_substeps},
            outputs=outputs,
            domain=gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
            mode=self._backend,
        )

    def _substep_stencil_set_inputs(
        self, stage, substep, timestep, state, stage_state, tmp_state, tendencies
    ):
        tendency_names = () if tendencies is None else tendencies.keys()

        self._dts.value = timestep.total_seconds() / self._substeps
        self._stage_substeps.value = (
            self._substeps
            if self.stages == 1
            else self.substep_fractions[stage] * self._substeps
        )

        for var in self._substep_output_properties:
            if substep == 0:
                self._substep_stencil_state_inputs[var][...] = state[var][...]
            self._substep_stencil_stage_state_inputs[var][...] = stage_state[var][...]
            self._substep_stencil_tmp_state_inputs[var][...] = (
                state[var][...] if substep == 0 else tmp_state[var][...]
            )
            if var in tendency_names:
                self._substep_stencil_tendencies_inputs[var][...] = tendencies[var][...]

    @staticmethod
    def _substep_stencil_defs(
        dts,
        substeps,
        s=None,
        stage_s=None,
        tmp_s=None,
        tnd_s=None,
        su=None,
        stage_su=None,
        tmp_su=None,
        tnd_su=None,
        sv=None,
        stage_sv=None,
        tmp_sv=None,
        tnd_sv=None,
        qv=None,
        stage_qv=None,
        tmp_qv=None,
        tnd_qv=None,
        qc=None,
        stage_qc=None,
        tmp_qc=None,
        tnd_qc=None,
        qr=None,
        stage_qr=None,
        tmp_qr=None,
        tnd_qr=None,
    ):
        i = gt.Index()
        j = gt.Index()
        k = gt.Index()

        outs = []

        if s is not None:
            out_s = gt.Equation()

            if tnd_s is None:
                out_s[i, j, k] = (
                    tmp_s[i, j, k] + (stage_s[i, j, k] - s[i, j, k]) / substeps
                )
            else:
                out_s[i, j, k] = (
                    tmp_s[i, j, k]
                    + (stage_s[i, j, k] - s[i, j, k]) / substeps
                    + dts * tnd_s[i, j, k]
                )

            outs.append(out_s)

        if su is not None:
            out_su = gt.Equation()

            if tnd_su is None:
                out_su[i, j, k] = (
                    tmp_su[i, j, k] + (stage_su[i, j, k] - su[i, j, k]) / substeps
                )
            else:
                out_su[i, j, k] = (
                    tmp_su[i, j, k]
                    + (stage_su[i, j, k] - su[i, j, k]) / substeps
                    + dts * tnd_su[i, j, k]
                )

            outs.append(out_su)

        if sv is not None:
            out_sv = gt.Equation()

            if tnd_sv is None:
                out_sv[i, j, k] = (
                    tmp_sv[i, j, k] + (stage_sv[i, j, k] - sv[i, j, k]) / substeps
                )
            else:
                out_sv[i, j, k] = (
                    tmp_sv[i, j, k]
                    + (stage_sv[i, j, k] - sv[i, j, k]) / substeps
                    + dts * tnd_sv[i, j, k]
                )

            outs.append(out_sv)

        if qv is not None:
            out_qv = gt.Equation()

            if tnd_qv is None:
                out_qv[i, j, k] = (
                    tmp_qv[i, j, k] + (stage_qv[i, j, k] - qv[i, j, k]) / substeps
                )
            else:
                out_qv[i, j, k] = (
                    tmp_qv[i, j, k]
                    + (stage_qv[i, j, k] - qv[i, j, k]) / substeps
                    + dts * tnd_qv[i, j, k]
                )

            outs.append(out_qv)

        if qc is not None:
            out_qc = gt.Equation()

            if tnd_qc is None:
                out_qc[i, j, k] = (
                    tmp_qc[i, j, k] + (stage_qc[i, j, k] - qc[i, j, k]) / substeps
                )
            else:
                out_qc[i, j, k] = (
                    tmp_qc[i, j, k]
                    + (stage_qc[i, j, k] - qc[i, j, k]) / substeps
                    + dts * tnd_qc[i, j, k]
                )

            outs.append(out_qc)

        if qr is not None:
            out_qr = gt.Equation()

            if tnd_qr is None:
                out_qr[i, j, k] = (
                    tmp_qr[i, j, k] + (stage_qr[i, j, k] - qr[i, j, k]) / substeps
                )
            else:
                out_qr[i, j, k] = (
                    tmp_qr[i, j, k]
                    + (stage_qr[i, j, k] - qr[i, j, k]) / substeps
                    + dts * tnd_qr[i, j, k]
                )

            outs.append(out_qr)

        return (*outs,)
