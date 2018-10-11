"""
This module contains:
	IsentropicPrognostic
"""
import abc
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.dynamics.diagnostics import WaterConstituent
from tasmania.dynamics.isentropic_fluxes import HorizontalIsentropicFlux, \
												VerticalIsentropicFlux
from tasmania.dynamics.sedimentation_flux import SedimentationFlux
from tasmania.physics.microphysics import RaindropFallVelocity
from tasmania.utils.data_utils import get_physical_constants

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class IsentropicPrognostic:
	"""
	Abstract base class whose derived classes implement different
	schemes to carry out the prognostic steps of the three-dimensional
	moist isentropic dynamical core. The conservative form of the
	governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'density_of_liquid_water':
			DataArray(1e3, attrs={'units': 'kg m^-3'}),
	}

	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		diagnostics : obj
			Instance of :class:`~tasmania.dynamics.diagnostics.IsentropicDiagnostics`
			calculating the diagnostic variables.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
			This is modified in-place by setting the number of boundary layers.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
			for the complete list of the available options.
		adiabatic_flow : `bool`, optional
			:obj:`True` for an adiabatic atmosphere, in which the potential temperature
			is conserved, :obj:`False` otherwise. Defaults to :obj:`True`.
		vertical_flux_scheme : `str`, optional
			String specifying the numerical vertical flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.VerticalIsentropicFlux`
			for the complete list of the available options. Defaults to :obj:`None`.
		sedimentation_on : `bool`, optional
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		sedimentation_flux_scheme : `str`, optional
			String specifying the method to use to compute the numerical
			sedimentation flux. See
			:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
			for the complete list of available options. Defaults to :obj:`None`.
		sedimentation_substeps : `int`, optional
			Number of sub-timesteps to perform in order to integrate the
			sedimentation flux. Defaults to 2.
		raindrop_fall_velocity_diagnostic : `obj`
			Instance of a :class:`sympl.Diagnostic` calculating the raindrop
			fall velocity. Defaults to :obj:`None`.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
        physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'density_of_liquid_water', in units compatible with \
					[kg m^-3].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic._d_physical_constants`
			for the default values.
		"""
		# Keep track of the input parameters
		self._grid                         = grid
		self._moist_on                     = moist_on
		self._diagnostics				   = diagnostics
		self._hboundary					   = horizontal_boundary_conditions
		self._hflux_scheme                 = horizontal_flux_scheme
		self._adiabatic_flow 			   = adiabatic_flow
		self._vflux_scheme                 = vertical_flux_scheme
		self._sedimentation_on             = sedimentation_on
		self._sedimentation_flux_scheme    = sedimentation_flux_scheme
		self._sedimentation_substeps       = sedimentation_substeps
		self._fall_velocity_diagnostic	   = raindrop_fall_velocity_diagnostic \
											 if raindrop_fall_velocity_diagnostic is not None \
											 else RaindropFallVelocity(grid, backend)
		self._backend                      = backend
		self._dtype						   = dtype

		# Set physical parameters values
		self._physical_constants = get_physical_constants(self._d_physical_constants,
														  physical_constants)

		# Instantiate the classes computing the numerical horizontal and vertical fluxes
		self._hflux = HorizontalIsentropicFlux.factory(self._hflux_scheme, grid, moist_on)
		self._hboundary.nb = self._hflux.nb
		if not adiabatic_flow:
			self._vflux = VerticalIsentropicFlux.factory(self._vflux_scheme, grid, moist_on)

		# Instantiate the classes computing the vertical derivative of the sedimentation flux
		# and diagnosing the mass fraction of precipitation water
		if sedimentation_on:
			self._sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
			self._water_constituent_diagnostic = WaterConstituent(grid, backend)

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
	def step_neglecting_vertical_motion(self, stage, dt, raw_state, raw_tendencies=None):
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
            	* air_pressure_on_interface_levels [Pa];
            	* isentropic_density_of_water_vapor [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_cloud_liquid_water [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_precipitation_water [kg m^-2 K^-1] (optional);
            	* montgomery_potential [m^2 s^-2];
            	* x_velocity_at_u_locations [m s^-1];
            	* x_momentum_isentropic [kg m^-1 K^-1 s^-1];
            	* y_velocity_at_v_locations [m s^-1];
            	* y_momentum_isentropic [kg m^-1 K^-1 s^-1].

		raw_tendencies : dict
            Dictionary whose keys are strings indicating tendencies,
            tendencies, and values are :class:`numpy.ndarray`\s containing
            the data for those tendencies.
            The dictionary may contain the following keys:

				* x_velocity [m s^-2];
				* y_velocity [m s^-2];
            	* mass_fraction_of_water_vapor_in_air [g g^-1 s^-1];
            	* mass_fraction_of_cloud_liquid_water_in_air [g g^-1 s^-1];
            	* mass_fraction_of_precipitation_water_in_air [g g^-1 s^-1].

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

	@abc.abstractmethod
	def step_integrating_vertical_advection(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time by integrating the vertical advection, i.e.,
		by accounting for the change over time in potential temperature.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		stage : int
			The stage to perform.
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		raw_state_now : dict
            Dictionary whose keys are strings indicating the model
            variables, and values are :class:`numpy.ndarray`\s containing
            the data for those variables at current time.
            The dictionary should contain the following keys:

            	* air_isentropic_density [kg m^-2 K^-1];
            	* air_pressure_on_interface_levels [Pa];
            	* isentropic_density_of_water_vapor [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_cloud_liquid_water [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_precipitation_water [kg m^-2 K^-1] (optional);
            	* x_velocity_at_u_locations [m s^-1];
            	* x_momentum_isentropic [kg m^-1 K^-1 s^-1];
            	* y_velocity_at_v_locations [m s^-1];
            	* y_momentum_isentropic [kg m^-1 K^-1 s^-1].

		raw_state_prv : obj
            Dictionary whose keys are strings indicating the model
            variables, and values are :class:`numpy.ndarray`\s containing
            the provisional data for those variables. Provisional data here
            mean the values obtained by sub-stepping the variables
            taking only the horizontal derivatives into account.
			The dictionary should contain the following keys:

            	* air_isentropic_density [kg m^-2 K^-1];
            	* isentropic_density_of_water_vapor [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_cloud_liquid_water [kg m^-2 K^-1] (optional);
            	* isentropic_density_of_precipitation_water [kg m^-2 K^-1] (optional);
            	* x_momentum_isentropic [kg m^-1 K^-1 s^-1];
            	* y_momentum_isentropic [kg m^-1 K^-1 s^-1].

			This may be the output of 
			:meth:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic.step_neglecting_vertical_motion`.

		Return
		------
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

	@abc.abstractmethod
	def step_integrating_sedimentation_flux(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the mass fraction of precipitation water by taking
		the sedimentation into account. For the sake of numerical stability,
		a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a time step which may be smaller than that specified by the user.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		stage : int
			The stage to perform.
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		raw_state_now : dict
            Dictionary whose keys are strings indicating the model
            variables, and values are :class:`np.ndarray`\s containing
            the data for those variables at current time.
            The dictionary should contain the following keys:

            	* accumulated_precipitation [mm] (optional);
            	* air_isentropic_density [kg m^-2 K^-1];
            	* air_pressure_on_interface_levels [Pa];
            	* mass_fraction_of_precipitation_water_in_air [g g^-1].

		raw_state_prv : obj
            Dictionary whose keys are strings indicating the model
            variables, and values are :class:`sympl.DataArray`\s containing
            the provisional data for those variables. Provisional data here
            mean the values obtained by sub-stepping the variables
            taking only the horizontal derivatives into account.
			The dictionary should contain the following keys:

            	* air_isentropic_density [kg m^-2 K^-1];
            	* mass_fraction_of_precipitation_water_in_air [g g^-1].

			This may be the output of
			:meth:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic.step_neglecting_vertical_motion`
			or
			:meth:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic.step_integrating_vertical_advection`.

		Return
		------
            Dictionary whose keys are strings indicating the conservative
            prognostic model variables, and values are :class:`sympl.DataArrays`\s
            containing the sub-stepped data for those variables.
            The dictionary contains the following keys:

            	* accumulated_precipitation [mm] (optional);
				* mass_fraction_of_precipitation_water_in_air [g g^-1];
				* precipitation [mm h^-1].
		"""

	@staticmethod
	def factory(scheme, grid, moist_on, diagnostics,
				horizontal_boundary_conditions, horizontal_flux_scheme,
				adiabatic_flow=True, vertical_flux_scheme=None,
				sedimentation_on=False, sedimentation_flux_scheme=None,
				sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Static method returning an instance of the derived class implementing
		the time stepping scheme specified by :data:`time_scheme`.

		Parameters
		----------
		scheme : str
			String specifying the time stepping method to implement. Either:

			* 'forward_euler', for the forward Euler scheme;
			* 'centered', for a centered scheme;
			* 'rk2', for the explicit two-stages, second-order Runge-Kutta scheme;
            * 'rk3cosmo', for the three-stages RK scheme as used in the
                `COSMO model <http://www.cosmo-model.org>`_; this method is
                nominally second-order, and third-order for linear problems;
			* 'rk3', for the explicit three-stages, third-order Runge-Kutta scheme.

		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		diagnostics : obj
			Instance of :class:`~tasmania.dynamics.diagnostics.IsentropicDiagnostics`
			calculating the diagnostic variables.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalFlux`
			for the complete list of the available options.
		adiabatic_flow : `bool`, optional
			:obj:`True` for an adiabatic atmosphere, in which the potential temperature
			is conserved, :obj:`False` otherwise. Defaults to :obj:`True`.
		vertical_flux_scheme : `str`, optional
			String specifying the numerical vertical flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.VerticalFlux`
			for the complete list of the available options. Defaults to :obj:`None`.
		sedimentation_on : `bool`, optional
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		sedimentation_flux_scheme : `str`, optional
			String specifying the method to use to compute the numerical
			sedimentation flux. See
			:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
			for the complete list of available options. Defaults to :obj:`None`.
		sedimentation_substeps : `int`, optional
			Number of sub-timesteps to perform in order to integrate the
			sedimentation flux. Defaults to 2.
		raindrop_fall_velocity_diagnostic : `obj`
			Instance of a :class:`sympl.Diagnostic` calculating the raindrop
			fall velocity. Defaults to :obj:`None`.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
        physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'density_of_liquid_water', in units compatible with \
					[kg m^-3].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic._d_physical_constants`
			for the default values.

		Return
		------
		obj :
			An instance of the derived class implementing the scheme specified
			by :data:`scheme`.
		"""
		import tasmania.dynamics._isentropic_prognostic as module

		arg_list = [grid, moist_on, diagnostics, horizontal_boundary_conditions,
					horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
					sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
					raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants]

		if scheme == 'forward_euler':
			return module._ForwardEuler(*arg_list)
		elif scheme == 'centered':
			return module._Centered(*arg_list)
		elif scheme == 'rk2':
			return module._RK2(*arg_list)
		elif scheme == 'rk3cosmo':
			return module._RK3COSMO(*arg_list)
		elif scheme == 'rk3':
			return module._RK3(*arg_list)
		else:
			raise ValueError('Unknown time integration scheme {}.\n'
							 'Available options: forward_euler, centered, rk2, rk3cosmo, rk3.'
							 .format(scheme))

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype
		raw_tendencies = {} if raw_tendencies is None else raw_tendencies
		tendency_names = raw_tendencies.keys()

		# Instantiate a GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Determine the size of the arrays which will serve as stencils'
		# inputs and outputs. These arrays may be shared with the stencil
		# in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the Numpy arrays which will serve as stencils' inputs
		# and which may be shared with the stencil in charge of integrating
		# the vertical advection
		self._in_s = np.zeros((li, lj, nz), dtype=dtype)
		self._in_su = np.zeros((li, lj, nz), dtype=dtype)
		self._in_sv = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv = np.zeros((li, lj, nz), dtype=dtype)
			self._in_sqc = np.zeros((li, lj, nz), dtype=dtype)

			# The array which will store the input mass fraction of
			# precipitation water may be shared either with stencil in
			# charge of integrating the vertical advection, or the stencil
			# taking care of sedimentation
			li = mi if (not self._sedimentation_on and self._adiabatic_flow) \
				 else max(mi, nx)
			lj = mj if (not self._sedimentation_on and self._adiabatic_flow) \
				 else max(mj, ny)
			self._in_sqr = np.zeros((li, lj, nz), dtype=dtype)

		# Allocate the input Numpy arrays not shared with any other stencil
		self._in_u   = np.zeros((mi+1,   mj, nz), dtype=dtype)
		self._in_v   = np.zeros((  mi, mj+1, nz), dtype=dtype)
		self._in_mtg = np.zeros((  mi,   mj, nz), dtype=dtype)
		if tendency_names is not None:
			if 'x_velocity' in tendency_names:
				self._in_u_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'y_velocity' in tendency_names:
				self._in_v_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'mass_fraction_of_water_vapor_in_air' in tendency_names:
				self._in_qv_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'mass_fraction_of_cloud_liquid_water_in_air' in tendency_names:
				self._in_qc_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'mass_fraction_of_precipitation_water_in_air' in tendency_names:
				self._in_qr_tnd = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_outputs(self):
		"""
		Allocate the Numpy arrays which will serve as outputs for
		the GT4Py stencils stepping the solution by neglecting any
		vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Determine the size of the output arrays; these arrays may be shared
		# with the stencil in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)
		nz = self._grid.nz

		# Allocate the output Numpy arrays which will serve as stencil's
		# outputs; these may be shared with the stencil in charge
		# of integrating the vertical advection
		self._out_s  = np.zeros((li, lj, nz), dtype=dtype)
		self._out_su = np.zeros((li, lj, nz), dtype=dtype)
		self._out_sv = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._out_sqv = np.zeros((li, lj, nz), dtype=dtype)
			self._out_sqc = np.zeros((li, lj, nz), dtype=dtype)

			# The array which will store the output mass fraction of precipitation
			# water may be shared either with stencil in charge of integrating the
			# vertical advection, or the stencil taking care of sedimentation
			li = mi if (not self._sedimentation_on and self._adiabatic_flow) \
				 else max(mi, nx)
			lj = mj if (not self._sedimentation_on and self._adiabatic_flow) \
				 else max(mj, ny)
			self._out_sqr = np.zeros((li, lj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_set_inputs(
		self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			u_tnd_on  = raw_tendencies.get('x_velocity', None) is not None
			v_tnd_on  = raw_tendencies.get('y_velocity', None) is not None
			qv_tnd_on = raw_tendencies.get('mass_fraction_of_water_vapor_in_air', None) \
						is not None
			qc_tnd_on = raw_tendencies.get('mass_fraction_of_cloud_liquid_water_in_air', None) \
						is not None
			qr_tnd_on = raw_tendencies.get('mass_fraction_of_precipitation_water_in_air', None) \
						is not None
		else:
			u_tnd_on = v_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = dt.total_seconds()

		# Extract the Numpy arrays representing the current solution
		s   = raw_state['air_isentropic_density']
		u   = raw_state['x_velocity_at_u_locations']
		v   = raw_state['y_velocity_at_v_locations']
		mtg = raw_state['montgomery_potential']
		su  = raw_state['x_momentum_isentropic']
		sv  = raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']
		if u_tnd_on:
			u_tnd = raw_tendencies['x_velocity']
		if v_tnd_on:
			v_tnd = raw_tendencies['y_velocity']
		if qv_tnd_on:
			qv_tnd = raw_tendencies['mass_fraction_of_water_vapor_in_air']
		if qc_tnd_on:
			qc_tnd = raw_tendencies['mass_fraction_of_cloud_liquid_water_in_air']
		if qr_tnd_on:
			qr_tnd = raw_tendencies['mass_fraction_of_precipitation_water_in_air']
		
		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s  [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(s)
		self._in_u  [:mi+1,   :mj, :] = self._hboundary.from_physical_to_computational_domain(u)
		self._in_v  [  :mi, :mj+1, :] = self._hboundary.from_physical_to_computational_domain(v)
		self._in_mtg[  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(mtg)
		self._in_su [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(su)
		self._in_sv [  :mi,   :mj, :] = self._hboundary.from_physical_to_computational_domain(sv)
		if self._moist_on:
			self._in_sqv[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqv)
			self._in_sqc[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqc)
			self._in_sqr[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sqr)
		if u_tnd_on:
			self._in_u_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u_tnd)
		if v_tnd_on:
			self._in_v_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(v_tnd)
		if qv_tnd_on:
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)
