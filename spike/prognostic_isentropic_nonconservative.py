# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import abc
import numpy as np

import gridtools as gt
from tasmania.dycore.flux_isentropic_nonconservative import (
    FluxIsentropicNonconservative,
)
from tasmania.dycore.flux_sedimentation import FluxSedimentation
from tasmania.conf import datatype
from tasmania.storages.state_isentropic import StateIsentropic
import python.utils.utils as utils


class PrognosticIsentropicNonconservative:
    """
    Abstract base class whose derived classes implement different schemes to carry out the
    prognostic steps of the three-dimensional moist isentropic_prognostic dynamical core.
    The nonconservative form of the governing equations is used.
    """

    # Make the class abstract
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        flux_scheme,
        grid,
        moist_on,
        backend,
        physics_dynamics_coupling_on,
        sedimentation_on,
    ):
        """
        Constructor.

        Parameters
        ----------
        flux_scheme : str
                String specifying the flux scheme to use. Either:

                * 'centered', for a second-order centered flux.

        grid : obj
                :class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
        moist_on : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        backend : obj
                :class:`gridtools.mode` specifying the backend for the GT4Py stencils.
        physics_dynamics_coupling_on : bool
                :obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential temperature,
                :obj:`False` otherwise.
        sedimentation_on : bool
                :obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
        """
        # Keep track of the input parameters
        self._flux_scheme, self._grid, self._moist_on, self._backend = (
            flux_scheme,
            grid,
            moist_on,
            backend,
        )
        self._physics_dynamics_coupling_on, self._sedimentation_on = (
            physics_dynamics_coupling_on,
            sedimentation_on,
        )

        # Instantiate the class computing the numerical horizontal and vertical fluxes
        self._flux = FluxIsentropicNonconservative.factory(
            flux_scheme, grid, moist_on
        )

        # Instantiate the class computing the vertical derivative of the sedimentation flux
        if sedimentation_on:
            self._flux_sedimentation = FluxSedimentation.factory(
                self._flux.order
            )

        # Initialize the attributes representing the diagnostic step and the lateral boundary conditions
        # Remark: these should be suitably set before calling the stepping method for the first time
        self._diagnostic, self._boundary = None, None

        # Initialize the attribute taking care of microphysics
        self._microphysics = None

        # Initialize the pointer to the compute function of the stencil in charge of coupling physics with dynamics
        # This will be properly re-directed the first time the corresponding forward method is invoked
        self._stencil_stepping_by_coupling_physics_with_dynamics = None

    @property
    def diagnostic(self):
        """
        Get the attribute implementing the diagnostic steps of the three-dimensional moist isentropic_prognostic dynamical core.
        If this is set to :obj:`None`, a :class:`ValueError` is thrown.

        Return
        ------
        obj :
                :class:`~tasmania.dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the
                three-dimensional moist isentropic_prognostic dynamical core.
        """
        if self._diagnostic is None:
            raise ValueError(
                """The attribute which is supposed to implement the diagnostic step of the moist isentroic """
                """dynamical core has not been previously set. Please set it correctly."""
            )
        return self._diagnostic

    @diagnostic.setter
    def diagnostic(self, value):
        """
        Set the attribute implementing the diagnostic steps of the three-dimensional moist isentropic_prognostic dynamical core.

        Parameter
        ---------
        value : obj
                :class:`~tasmania.dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the
                three-dimensional moist isentropic_prognostic dynamical core.
        """
        self._diagnostic = value

    @property
    def boundary(self):
        """
        Get the attribute implementing the horizontal boundary conditions.
        If this is set to :obj:`None`, a :class:`ValueError` is thrown.

        Return
        ------
        obj :
                Instance of the derived class of :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` implementing
                the horizontal boundary conditions.
        """
        if self._boundary is None:
            raise ValueError(
                """The attribute which is supposed to implement the horizontal boundary conditions """
                """has not been previously set. Please set it correctly."""
            )
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        """
        Set the attribute implementing the horizontal boundary conditions.

        Parameter
        ---------
        value : obj
                Instance of the derived class of :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` implementing the
                horizontal boundary conditions.
        """
        self._boundary = value

    @property
    def nb(self):
        """
        Get the number of lateral boundary layers.

        Return
        ------
        int :
                The number of lateral boundary layers.
        """
        return self._flux.nb

    @property
    def microphysics(self):
        """
        Get the attribute taking care of microphysics.
        If this is set to :obj:`None`, a :class:`ValueError` is thrown.

        Return
        ------
        obj :
                Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
                or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics`.
        """
        if self._microphysics is None:
            return ValueError(
                """The attribute which is supposed to take care of the microphysical dynamics """
                """has not been previously set. Please set it correctly."""
            )
        return self._microphysics

    @microphysics.setter
    def microphysics(self, micro):
        """
        Set the attribute taking care of microphysics.

        Parameters
        ----------
        micro : obj
                Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
                or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics`.
        """
        self._microphysics = micro

    @abc.abstractmethod
    def step_neglecting_vertical_advection(
        self, dt, state, state_old=None, diagnostics=None, tendencies=None
    ):
        """
        Method advancing the prognostic model variables one time step forward.
        Only horizontal derivates are considered; possible vertical derivatives are disregarded.
        As this method is marked as abstract, its implementation is delegated to the derived classes.

        Parameters
        ----------
        dt : obj
                :class:`datetime.timedelta` representing the time step.
        state : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
                * montgomery_potential (isentropic_prognostic);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

        state_old : `obj`, optional
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the old state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

        diagnostics : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` possibly storing diagnostics.
                For the time being, this is not actually used.
        tendencies : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` possibly storing tendencies.
                For the time being, this is not actually used.

        Return
        ------
        obj :
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
        """

    def step_coupling_physics_with_dynamics(
        self, dt, state_now, state_prv, diagnostics
    ):
        """
        Method advancing the prognostic model variables one time step forward by coupling physics with
        dynamics, i.e., by accounting for the change over time in potential temperature.

        Parameters
        ----------
        dt : obj
                :class:`datetime.timedelta` representing the time step.
        state_now : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

        state_prv : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
                the state stepped taking only the horizontal derivatives into account.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

                This may be the output of
                :meth:`~tasmania.dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative.step_neglecting_vertical_advection`.
        diagnostics : obj
                :class:`~tasmania.storages.grid_data.GridData` collecting the following variables:

                * change_over_time_in_air_potential_temperature (unstaggered).

        Return
        ------
        obj :
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
        """
        # Initialize the output state
        time_now = utils.convert_datetime64_to_datetime(
            state_now["air_isentropic_density"].coords["time"].values[0]
        )
        state_new = StateIsentropic(time_now + dt, self._grid)

        # The first time this method is invoked, initialize the GT4Py stencil
        if self._stencil_stepping_by_coupling_physics_with_dynamics is None:
            self._stencil_stepping_by_coupling_physics_with_dynamics_initialize(
                state_now
            )

        # Set stencil's inputs
        self._stencil_stepping_by_coupling_physics_with_dynamics_set_inputs(
            dt, state_now, state_prv
        )

        # Run the stencil
        self._stencil_stepping_by_coupling_physics_with_dynamics.compute()

        # Set the lower and upper layers
        nb = self._flux.nb
        self._out_s[:, :, :nb], self._out_s[:, :, -nb:] = (
            self._in_s_prv[:, :, :nb],
            self._in_s_prv[:, :, -nb:],
        )
        self._out_u[:, :, :nb], self._out_u[:, :, -nb:] = (
            self._in_u_prv[:, :, :nb],
            self._in_u_prv[:, :, -nb:],
        )
        self._out_v[:, :, :nb], self._out_v[:, :, -nb:] = (
            self._in_v_prv[:, :, :nb],
            self._in_v_prv[:, :, -nb:],
        )
        if self._moist_on:
            self._out_qv[:, :, :nb], self._out_qv[:, :, -nb:] = (
                self._in_qv_prv[:, :, :nb],
                self._in_qv_prv[:, :, -nb:],
            )
            self._out_qc[:, :, :nb], self._out_qc[:, :, -nb:] = (
                self._in_qc_prv[:, :, :nb],
                self._in_qc_prv[:, :, -nb:],
            )
            self._out_qr[:, :, :nb], self._out_qr[:, :, -nb:] = (
                self._in_qr_prv[:, :, :nb],
                self._in_qr_prv[:, :, -nb:],
            )

        # Update the output state
        state_new.add_variables(
            time_now + dt,
            air_isentropic_density=self._out_s,
            x_velocity=self._out_u,
            y_velocity=self._out_v,
        )
        if self._moist_on:
            state_new.add_variables(
                time_now + dt,
                mass_fraction_of_water_vapor_in_air=self._out_qv,
                mass_fraction_of_cloud_liquid_water_in_air=self._out_qc,
                mass_fraction_of_precipitation_water_in_air=self._out_qr,
            )

        return state_new

    @abc.abstractmethod
    def step_integrating_sedimentation_flux(
        self, dt, state_now, state_prv, diagnostics=None
    ):
        """
        Method advancing the mass fraction of precipitation water by taking the sedimentation into account.
        For the sake of numerical stability, a time-splitting strategy is pursued, i.e., the sedimentation is
        integrated using a timestep which may be smaller than that specified by the user.
        As this method is marked as abstract, its implementation is delegated to the derived classes.

        Parameters
        ----------
        dt : obj
                :class:`datetime.timedelta` representing the time step.
        state_now : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * height or height_on_interface_levels (:math:`z`-staggered);
                * mass_fraction_of_precipitation_water_in air (unstaggered).

        state_prv : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
                the state stepped without taking the sedimentation flux into account.
                It should contain the following variables:

                * mass_fraction_of_precipitation_water_in_air (unstaggered).

                This may be the output of either
                :meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection` or
                :meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_coupling_physics_with_dynamics`.
        diagnostics : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` collecting the following diagnostics:

                * accumulated_precipitation (unstaggered, two-dimensional);
                * precipitation (unstaggered, two-dimensional).

        Returns
        -------
        state_new : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the following updated variables:

                * mass_fraction_of_precipitation_water_in air (unstaggered).

        diagnostics_out : obj
                :class:`~tasmania.storages.grid_data.GridData` collecting the output diagnostics, i.e.:

                * accumulated_precipitation (unstaggered, two-dimensional);
                * precipitation (unstaggered, two-dimensional).
        """

    @staticmethod
    def factory(
        time_scheme,
        flux_scheme,
        grid,
        moist_on,
        backend,
        physics_dynamics_coupling_on,
        sedimentation_on,
    ):
        """
        Static method returning an instace of the derived class implementing the time stepping scheme specified
        by :data:`time_scheme`, using the flux scheme specified by :data:`flux_scheme`.

        Parameters
        ----------
        time_scheme : str
                String specifying the time stepping method to implement. Either:

                * 'centered', for a centered scheme.

        flux_scheme : str
                String specifying the scheme to use. Either:

                * 'centered', for a second-order centered flux.

        grid : obj
                :class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
        moist_on : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        backend : obj
                :class:`gridtools.Mode` specifying the backend for the GT4Py stencils.
        physics_dynamics_coupling_on : bool
                :obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential temperature,
                :obj:`False` otherwise.
        sedimentation_on : bool
                :obj:`True` to account for rain sedimentation, :obj:`False` otherwise.

        Return
        ------
        obj :
                An instace of the derived class implementing the scheme specified by :data:`scheme`.
        """
        if time_scheme == "centered":
            from tasmania.dycore.prognostic_isentropic_nonconservative_centered import (
                PrognosticIsentropicNonconservativeCentered,
            )

            return PrognosticIsentropicNonconservativeCentered(
                flux_scheme,
                grid,
                moist_on,
                backend,
                physics_dynamics_coupling_on,
                sedimentation_on,
            )
        else:
            raise ValueError("Unknown time integration scheme.")

    def _stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(
        self, mi, mj
    ):
        """
        Allocate the attributes which serve as inputs to the GT4Py stencils which step the solution
        disregarding the vertical advection.

        Parameters
        ----------
        mi : int
                :math:`x`-extent of an input array representing an :math:`x`-unstaggered field.
        mj : int
                :math:`y`-extent of an input array representing a :math:`y`-unstaggered field.
        """
        # Shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # Keep track of the input arguments
        self._mi, self._mj = mi, mj

        # Instantiate a GT4Py Global representing the timestep
        self._dt = gt.Global()

        # Determine the size of the input arrays
        # These arrays may be shared with the stencil in charge of coupling physics with dynamics
        li = mi if not self._physics_dynamics_coupling_on else max(mi, nx)
        lj = mj if not self._physics_dynamics_coupling_on else max(mj, ny)

        # Allocate the input Numpy arrays which may be shared with the stencil in charge of coupling physics with dynamics
        self._in_s = np.zeros((li, lj, nz), dtype=datatype)
        self._in_u = np.zeros((li + 1, lj, nz), dtype=datatype)
        self._in_v = np.zeros((li, lj + 1, nz), dtype=datatype)
        if self._moist_on:
            self._in_qv = np.zeros((li, lj, nz), dtype=datatype)
            self._in_qc = np.zeros((li, lj, nz), dtype=datatype)

            # The array which will store the input mass fraction of precipitation water may be shared
            # either with stencil in charge of coupling physics with dynamics, or the stencil taking care of sedimentation
            li = (
                mi
                if not (
                    self._sedimentation_on
                    and self._physics_dynamics_coupling_on
                )
                else max(mi, nx)
            )
            lj = (
                mj
                if not (
                    self._sedimentation_on
                    and self._physics_dynamics_coupling_on
                )
                else max(mj, ny)
            )
            self._in_qr = np.zeros((li, lj, nz), dtype=datatype)

        # Allocate the input Numpy arrays not shared with any other stencil
        self._in_mtg = np.zeros((mi, mj, nz), dtype=datatype)

    def _stencils_stepping_by_neglecting_vertical_advection_allocate_outputs(
        self, mi, mj
    ):
        """
        Allocate the Numpy arrays which will store the solution updated by neglecting the vertical advection.

        Parameters
        ----------
        mi : int
                :math:`x`-extent of an output array representing an :math:`x`-unstaggered field.
        mj : int
                :math:`y`-extent of an output array representing a :math:`y`-unstaggered field.
        """
        # Keep track of the input arguments
        self._mi, self._mj = mi, mj

        # Determine the size of the output arrays
        # These arrays may be shared with the stencil in charge of coupling physics with dynamics
        li = mi if not self._physics_dynamics_coupling_on else max(mi, nx)
        lj = mj if not self._physics_dynamics_coupling_on else max(mj, ny)
        nz = self._grid.nz

        # Allocate the output Numpy arrays which may be shared with the stencil in charge of coupling physics with dynamics
        self._out_s = np.zeros((li, lj, nz), dtype=datatype)
        self._out_u = np.zeros((li + 1, lj, nz), dtype=datatype)
        self._out_v = np.zeros((li, lj + 1, nz), dtype=datatype)
        if self._moist_on:
            self._out_qv = np.zeros((li, lj, nz), dtype=datatype)
            self._out_qc = np.zeros((li, lj, nz), dtype=datatype)

            # The array which will store the output mass fraction of precipitation water may be shared
            # either with stencil in charge of coupling physics with dynamics, or the stencil taking care of sedimentation
            li = (
                mi
                if not (
                    self._sedimentation_on
                    and self._physics_dynamics_coupling_on
                )
                else max(mi, nx)
            )
            lj = (
                mj
                if not (
                    self._sedimentation_on
                    and self._physics_dynamics_coupling_on
                )
                else max(mj, ny)
            )
            self._out_qr = np.zeros((li, lj, nz), dtype=datatype)

    def _stencils_stepping_by_neglecting_vertical_advection_set_inputs(
        self, dt, state
    ):
        """
        Update the attributes which serve as inputs to the GT4Py stencils which step the solution
        disregarding the vertical advection.

        Parameters
        ----------
        dt : obj
                A :class:`datetime.timedelta` representing the time step.
        state : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * montgomery_potential (isentropic_prognostic);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
        """
        # Shortcuts
        mi, mj = self._mi, self._mj

        # Update the local time step
        self._dt.value = (
            1.0e-6 * dt.microseconds if dt.seconds == 0.0 else dt.seconds
        )

        # Extract the Numpy arrays representing the current solution
        s = state["air_isentropic_density"].values[:, :, :, 0]
        u = state["x_velocity"].values[:, :, :, 0]
        v = state["y_velocity"].values[:, :, :, 0]
        mtg = state["montgomery_potential"].values[:, :, :, 0]
        if self._moist_on:
            qv = state["mass_fraction_of_water_vapor_in_air"].values[
                :, :, :, 0
            ]
            qc = state["mass_fraction_of_cloud_liquid_water_in_air"].values[
                :, :, :, 0
            ]
            qr = state["mass_fraction_of_precipitation_water_in_air"].values[
                :, :, :, 0
            ]

        # Update the Numpy arrays which serve as inputs to the GT4Py stencils
        self._in_s[
            :mi, :mj, :
        ] = self.boundary.from_physical_to_computational_domain(s)
        self._in_u[
            : mi + 1, :mj, :
        ] = self.boundary.from_physical_to_computational_domain(u)
        self._in_v[
            :mi, : mj + 1, :
        ] = self.boundary.from_physical_to_computational_domain(v)
        self._in_mtg[
            :mi, :mj, :
        ] = self.boundary.from_physical_to_computational_domain(mtg)
        if self._moist_on:
            self._in_qv[
                :mi, :mj, :
            ] = self.boundary.from_physical_to_computational_domain(qv)
            self._in_qc[
                :mi, :mj, :
            ] = self.boundary.from_physical_to_computational_domain(qc)
            self._in_qr[
                :mi, :mj, :
            ] = self.boundary.from_physical_to_computational_domain(qr)

    def _stencil_stepping_by_coupling_physics_with_dynamics_initialize(
        self, state_now
    ):
        """
        Initialize the GT4Py stencil in charge of stepping the solution by coupling physics with dynamics,
        i.e., by accounting for the change over time in potential temperature.

        Parameters
        ----------
        state_now : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered).
        """
        ### TODO ###

    def _stencil_stepping_by_coupling_physics_with_dynamics_allocate_inputs(
        self,
    ):
        """
        Allocate the attributes which serve as inputs to the GT4Py stencil which step the solution
        by coupling physics with dynamics, i.e., accounting for the change over time in potential temperature.
        """
        ### TODO ###

    def _stencil_stepping_by_coupling_physics_with_dynamics_allocate_outputs(
        self,
    ):
        """
        Allocate the Numpy arrays which will store the solution updated by coupling physics with dynamics.
        """
        ### TODO ###

    def _stencil_stepping_by_coupling_physics_with_dynamics_set_inputs(
        self, dt, state_now, state_prv, diagnostics
    ):
        """
        Update the attributes which serve as inputs to the GT4Py stencil which steps the solution
        by integrating the vertical advection, i.e., by accounting for the change over time in potential temperature.

        Parameters
        ----------
        dt : obj
                A :class:`datetime.timedelta` representing the time step.
        state_now : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

        state_prv : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
                the state stepped taking only the horizontal derivatives into account.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

                This may be the output of
                :meth:`~tasmania.dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative.step_neglecting_vertical_advection`.
        diagnostics : obj
                :class:`~tasmania.storages.grid_data.GridData` collecting the following variables:

                * change_over_time_in_air_potential_temperature (unstaggered).
        """
        ### TODO ###

    @abc.abstractmethod
    def _stencil_stepping_by_coupling_physics_with_dynamics_defs(
        dt,
        in_w,
        in_s_now,
        in_s_prv,
        in_u_now,
        in_u_prv,
        in_v_now,
        in_v_prv,
        qv_now=None,
        qv_prv=None,
        qc_now=None,
        qc_prv=None,
        qr_now=None,
        qr_prv=None,
    ):
        """
        GT4Py stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
        change over time in potential temperature.
        As this method is marked as abstract, its implementation is delegated to the derived classes.

        Parameters
        ----------
        dt : obj
                :class:`gridtools.Global` representing the time step.
        in_w : array_like
                :class:`numpy.ndarray` representing the vertical velocity, i.e., the change over time in potential temperature.
        in_s_now : obj
                :class:`gridtools.Equation` representing the current isentropic_prognostic density.
        in_s_prv : obj
                :class:`gridtools.Equation` representing the provisional isentropic_prognostic density.
        in_u_now : obj
                :class:`gridtools.Equation` representing the current :math:`x`-velocity.
        in_u_prv : obj
                :class:`gridtools.Equation` representing the provisional :math:`x`-velocity.
        in_v_now : obj
                :class:`gridtools.Equation` representing the current :math:`y`-velocity.
        in_v_prv : obj
                :class:`gridtools.Equation` representing the provisional :math:`y`-velocity.
        in_qv_now : `obj`, optional
                :class:`gridtools.Equation` representing the current mass fraction of water vapor.
        in_qv_prv : `obj`, optional
                :class:`gridtools.Equation` representing the provisional mass fraction of water vapor.
        in_qc_now : `obj`, optional
                :class:`gridtools.Equation` representing the current mass fraction of cloud liquid water.
        in_qc_prv : `obj`, optional
                :class:`gridtools.Equation` representing the provisional mass fraction of cloud liquid water.
        in_qr_now : `obj`, optional
                :class:`gridtools.Equation` representing the current mass fraction of precipitation water.
        in_qr_prv : `obj`, optional
                :class:`gridtools.Equation` representing the provisional mass fraction of precipitation water.

        Returns
        -------
        out_s : obj
                :class:`gridtools.Equation` representing the updated isentropic_prognostic density.
        out_u : obj
                :class:`gridtools.Equation` representing the updated :math:`x`-velocity.
        out_v : obj
                :class:`gridtools.Equation` representing the updated :math:`y`-velocity.
        out_qv : `obj`, optional
                :class:`gridtools.Equation` representing the updated mass fraction of water vapor.
        out_qc : `obj`, optional
                :class:`gridtools.Equation` representing the updated mass fraction of cloud liquid water.
        out_qr : `obj`, optional
                :class:`gridtools.Equation` representing the updated mass fraction of precipitation water.
        """
