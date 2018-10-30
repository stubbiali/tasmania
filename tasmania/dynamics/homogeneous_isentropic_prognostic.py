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
	HomogeneousIsentropicPrognostic
"""
import abc
import numpy as np

import gridtools as gt
from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


# Convenient aliases
mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


class HomogeneousIsentropicPrognostic:
	"""
	Abstract base class whose derived classes implement different
	schemes to carry out the prognostic steps of the three-dimensional
	homogeneous, moist, isentropic dynamical core. Here, _homogeneous_ means
	that the pressure gradient terms, i.e., the terms involving the gradient
	of the Montgomery potential, are not included in the dynamics, but
	rather parameterized. This holds also for any sedimentation motion.
	The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
			This is modified in-place by setting the number of boundary layers.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousIsentropicFlux`
			for the complete list of the available options.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Keep track of the input parameters
		self._grid          = grid
		self._moist_on      = moist_on
		self._hboundary		= horizontal_boundary_conditions
		self._hflux_scheme  = horizontal_flux_scheme
		self._backend       = backend
		self._dtype			= dtype

		# Instantiate the classes computing the numerical horizontal and vertical fluxes
		self._hflux = HorizontalHomogeneousIsentropicFlux.factory(self._hflux_scheme,
																  grid, moist_on)
		self._hboundary.nb = self._hflux.nb

	@property
	@abc.abstractmethod
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Return
		------
		int :
			The number of stages performed by the time-integration scheme.
		"""

	@property
	def nb(self):
		"""
		Return
		------
		int :
			The number of lateral boundary layers.
		"""
		return self._hflux.nb

	@property
	def horizontal_boundary(self):
		"""
		Return
		------
		obj :
			Object in charge of handling the lateral boundary conditions.
		"""
		return self._hboundary

	@abc.abstractmethod
	def __call__(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		stage : int
			Index of the stage to perform.
		dt : timedelta
			:class:`datetime.timedelta` representing the time step.
		raw_state : dict
            Dictionary whose keys are strings indicating the model
            variables, and values are :class:`numpy.ndarray`\s containing
            the data for those variables at current time.
            The dictionary should contain the following keys:

            	* air_isentropic_density [kg m^-2 K^-1];
            	* isentropic_density_of_water_vapor [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_cloud_liquid_water [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_precipitation_water [kg m^-2 K^-1] (optional);
            	* x_velocity_at_u_locations [m s^-1];
            	* x_momentum_isentropic [kg m^-1 K^-1 s^-1];
            	* y_velocity_at_v_locations [m s^-1];
            	* y_momentum_isentropic [kg m^-1 K^-1 s^-1].

		raw_tendencies : dict
            Dictionary whose keys are strings indicating tendencies,
            tendencies, and values are :class:`numpy.ndarray`\s containing
            the data for those tendencies.
            The dictionary may contain the following keys:

				* air_isentropic_density [kg m^-2 K^-1 s^-1];
            	* mass_fraction_of_water_vapor_in_air [g g^-1 s^-1];
            	* mass_fraction_of_cloud_liquid_water_in_air [g g^-1 s^-1];
            	* mass_fraction_of_precipitation_water_in_air [g g^-1 s^-1].
				* x_momentum_isentropic [kg m^-1 K^-1 s^-2];
				* y_momentum_isentropic [kg m^-1 K^-1 s^-2];

		Return
		------
		dict :
            Dictionary whose keys are strings indicating the conservative
            prognostic model variables, and values are :class:`numpy.ndarray`\s
            containing the sub-stepped data for those variables.
            The dictionary contains the following keys:

            	* air_isentropic_density [kg m^-2 K^-1];
            	* isentropic_density_of_water_vapor [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_cloud_liquid_water [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_precipitation_water [kg m^-2 K^-1] (optional);
            	* x_momentum_isentropic [kg m^-1 K^-1 s^-1];
            	* y_momentum_isentropic [kg m^-1 K^-1 s^-1].
		"""

	@staticmethod
	def factory(scheme, grid, moist_on, horizontal_boundary_conditions,
				horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Static method returning an instance of the derived class implementing
		the time stepping scheme specified by :data:`time_scheme`.

		Parameters
		----------
		scheme : str
			String specifying the time stepping method to implement. Either:

			* 'forward_euler', for the forward Euler scheme;
			* 'centered', for a centered scheme;
			* 'rk2', for the two-stages, second-order Runge-Kutta (RK) scheme;
            * 'rk3cosmo', for the three-stages RK scheme as used in the
                `COSMO model <http://www.cosmo-model.org>`_; this method is
                nominally second-order, and third-order for linear problems;
			* 'rk3', for the three-stages, third-order RK scheme.

		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousHorizontalFlux`
			for the complete list of the available options.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.

		Return
		------
		obj :
			An instance of the derived class implementing the scheme specified
			by :data:`scheme`.
		"""
		import tasmania.dynamics._homogeneous_isentropic_prognostic as module
		arg_list = [grid, moist_on, horizontal_boundary_conditions,
					horizontal_flux_scheme, backend, dtype]

		if scheme == 'forward_euler':
			return module.ForwardEuler(*arg_list)
		elif scheme == 'centered':
			return module.Centered(*arg_list)
		elif scheme == 'rk2':
			return module.RK2(*arg_list)
		elif scheme == 'rk3cosmo':
			return module.RK3COSMO(*arg_list)
		elif scheme == 'rk3':
			return module.RK3(*arg_list)
		else:
			raise ValueError('Unknown time integration scheme {}. Available options: '
							 'forward_euler, centered, rk2, rk3cosmo, rk3.'.format(scheme))

	def _stencil_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		nz = self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype
		raw_tendencies = {} if raw_tendencies is None else raw_tendencies
		tendency_names = raw_tendencies.keys()

		# Instantiate a GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will store the current solution
		# and serve as stencil's inputs
		self._in_s  = np.zeros((  mi,   mj, nz), dtype=dtype)
		self._in_u  = np.zeros((mi+1,   mj, nz), dtype=dtype)
		self._in_v  = np.zeros((  mi, mj+1, nz), dtype=dtype)
		self._in_su = np.zeros((  mi,   mj, nz), dtype=dtype)
		self._in_sv = np.zeros((  mi,   mj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqr = np.zeros((mi, mj, nz), dtype=dtype)

		# Allocate the input Numpy arrays which will store the tendencies
		# and serve as stencil's inputs
		if tendency_names is not None:
			if 'air_isentropic_density' in tendency_names:
				self._in_s_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mf_wv in tendency_names:
				self._in_qv_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mf_clw in tendency_names:
				self._in_qc_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mf_pw in tendency_names:
				self._in_qr_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'x_momentum_isentropic' in tendency_names:
				self._in_su_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'y_momentum_isentropic' in tendency_names:
				self._in_sv_tnd = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencil_allocate_outputs(self):
		"""
		Allocate the Numpy arrays which will serve as outputs for
		the GT4Py stencils stepping the solution by neglecting any
		vertical motion.
		"""
		# Shortcuts
		nz = self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Allocate the Numpy arrays which will serve as stencil's outputs
		self._out_s  = np.zeros((mi, mj, nz), dtype=dtype)
		self._out_su = np.zeros((mi, mj, nz), dtype=dtype)
		self._out_sv = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._out_sqv = np.zeros((mi, mj, nz), dtype=dtype)
			self._out_sqc = np.zeros((mi, mj, nz), dtype=dtype)
			self._out_sqr = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencil_set_inputs(self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			s_tnd_on  = raw_tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = raw_tendencies.get(mf_wv, None) is not None
			qc_tnd_on = raw_tendencies.get(mf_clw, None) is not None
			qr_tnd_on = raw_tendencies.get(mf_pw, None) is not None
			su_tnd_on = raw_tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = raw_tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = dt.total_seconds()

		# Extract the Numpy arrays representing the current solution
		s   = raw_state['air_isentropic_density']
		u   = raw_state['x_velocity_at_u_locations']
		v   = raw_state['y_velocity_at_v_locations']
		su  = raw_state['x_momentum_isentropic']
		sv  = raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']
		if s_tnd_on:
			s_tnd = raw_tendencies['air_isentropic_density']
		if qv_tnd_on:
			qv_tnd = raw_tendencies[mf_wv]
		if qc_tnd_on:
			qc_tnd = raw_tendencies[mf_clw]
		if qr_tnd_on:
			qr_tnd = raw_tendencies[mf_pw]
		if su_tnd_on:
			su_tnd = raw_tendencies['x_momentum_isentropic']
		if sv_tnd_on:
			sv_tnd = raw_tendencies['y_momentum_isentropic']

		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s  [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(s)
		self._in_u  [:mi+1,   :mj, :] = self._hboundary.from_physical_to_computational_domain(u)
		self._in_v  [  :mi, :mj+1, :] = self._hboundary.from_physical_to_computational_domain(v)
		self._in_su [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(su)
		self._in_sv [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(sv)
		if self._moist_on:
			self._in_sqv[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqv)
			self._in_sqc[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqc)
			self._in_sqr[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqr)
		if s_tnd_on:
			self._in_s_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s_tnd)
		if su_tnd_on:
			self._in_su_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su_tnd)
		if sv_tnd_on:
			self._in_sv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv_tnd)
		if qv_tnd_on:
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)
