import abc
import numpy as np

import gridtools as gt
from tasmania.namelist import cp, datatype, L, Rd, Rv
from tasmania.parameterizations.adjustments import AdjustmentMicrophysics
from tasmania.storages.grid_data import GridData
import tasmania.utils.utils as utils
import tasmania.utils.utils_meteo as utils_meteo

class AdjustmentMicrophysicsKesslerWRFSaturation(AdjustmentMicrophysics):
	"""
	This class inherits :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` 
	to implement the saturation adjustment as predicted by the WRF version of the Kessler scheme.

	Attributes
	----------
	inputs : tuple of str
		The variables required in the input state when the object is called.
	diagnostics : tuple of str
		The diagnostic variables output by the object.
	outputs : tuple of str
		The variables which gets adjusted by the object.
	input_properties : dict
		A dictionary whose keys are variables required in the state when the object is called, 
		and values are dictionaries which indicate 'dims' and 'units'.
	diagnostic_properties : dict
		 A dictionary whose keys are diagnostics output by the object,
		 and values are dictionaries which indicate 'dims' and 'units'.
	output_properties : dict
		A dictionary whose keys are variables adjusted when the object is called, 
		and values are dictionaries which indicate 'dims' and 'units'.
	"""
	# Set attributes inherited from sympl's Implicit
	inputs      = ('air_density'                               , 
				   'air_pressure'                              , 
				   'exner_function'                            , 
				   'air_temperature'                           ,
				   'mass_fraction_of_water_vapor_in_air'       , 
				   'mass_fraction_of_cloud_liquid_water_in_air')
	diagnostics = ('raindrop_fall_velocity')
	outputs     = ('mass_fraction_of_water_vapor_in_air'       , 
				   'mass_fraction_of_cloud_liquid_water_in_air')
	input_properties = {'air_density': 
							{'dims': '*',
							 'units': 'kg m-3'},
						'air_pressure': 
							{'dims': '*',
							 'units': 'Pa'},
						'exner_function': 
							{'dims': '*',
							 'units': 'm2 s-2 K-2',
							 'match_dims_like': ['air_pressure']},
						'air_temperature': 
							{'dims': '*',
							 'units': 'K',
							 'match_dims_like': ['air_density']}, 
						'mass_fraction_of_water_vapor_in_air': 
							{'dims': '*',
							 'units': 'kg kg-1',
							 'match_dims_like': ['air_density']},
						'mass_fraction_of_cloud_liquid_water_in_air': 
							{'dims': '*',
							 'units': 'kg kg-1',
							 'match_dims_like': ['air_density']}}
	diagnostic_properties = {'raindrop_fall_velocity': 
								{'dims': '*',
								 'units': 'm s-1'}}
	output_properties = {'mass_fraction_of_water_vapor_in_air': 
							{'dims': '*',
							 'units': 'kg kg-1'},
						 'mass_fraction_of_cloud_liquid_water_in_air': 
						 	{'dims': '*',
							 'units': 'kg kg-1'}}

	def __init__(self, grid, rain_evaporation_on, backend, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.

		Note
		----
		To instantiate this class, one should prefer the static method 
		:meth:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics.factory` of 
		:class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics`.
		"""
		# Call parent's constructor
		super().__init__(grid, rain_evaporation_on, backend)

		# Constants for Tetens formula
		self._p0 = 610.78
		self._alpha = 17.27
		self._Tr = 273.15
		self._bw = 35.85
		
		# Shortcuts
		self._beta   = Rd / Rv
		self._beta_c = 1. - self._beta
		self._kappa  = L * self._alpha * self._beta * (self._Tr - self._bw) / cp 

		# Initialize pointers to the underlying GT4Py stencils
		# They will be properly re-directed the first time the entry point method is invoked
		self._stencil_adjustment 			 = None
		self._stencil_raindrop_fall_velocity = None

	def __call__(self, state, dt):
		"""
		Entry-point method performing the saturation adjustment.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* air_pressure (unstaggered) or air_pressure_on_interface_levels (:math:`z`-staggered);
			* exner_function (unstaggered) or exner_function_on_interface_levels (:math:`z`-staggered);
			* air_temperature (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);

		dt : obj
			:class:`datetime.timedelta` representing the timestep.

		Returns
		-------
		diagnostics : obj
			Empty :class:`~storages.grid_data.GridData`, as no diagnostics are computed.
		state_new : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the adjusted state.
			It contains the following updated variables:

			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_adjustment is None:
			self._stencil_adjustment_initialize()

		# Update the local time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Extract the required model variables
		self._in_p[:,:,:]   = state['air_pressure'].values[:,:,:,0] if state['air_pressure'] is not None else \
							  0.5 * (state['air_pressure_on_interface_levels'].values[:,:,:-1,0] +
							  		 state['air_pressure_on_interface_levels'].values[:,:, 1:,0])
		#self._in_exn[:,:,:] = state['exner_function'].values[:,:,:,0]
		self._in_T[:,:,:]	= state['air_temperature'].values[:,:,:,0]
		self._in_qv[:,:,:]  = state['mass_fraction_of_water_vapor_in_air'].values[:,:,:,0]
		self._in_qc[:,:,:]  = state['mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0]

		# Compute the saturation water vapor pressure via Tetens' formula
		self._in_ps[:,:,:] = self._p0 * np.exp(self._alpha * (self._in_T[:,:,:] - self._Tr) / \
											   (self._in_T[:,:,:] - self._bw))

		# Run the stencil
		self._stencil_adjustment.compute()

		# Instantiate the output state
		time = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		state_new = type(state)(time, self._grid, 
								mass_fraction_of_water_vapor_in_air         = self._out_qv,
					  			mass_fraction_of_cloud_liquid_water_in_air  = self._out_qc)

		# Instantiate an empty GridData, acting as the output diagnostic
		diagnostics = GridData(time, self._grid)

		return diagnostics, state_new

	def get_raindrop_fall_velocity(self, state):
		"""
		Get the raindrop fall velocity.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).
			
		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the raindrop fall velocity.
		"""
		# If this is the first time this method is invoked, initialize the auxiliary GT4Py stencil
		if self._stencil_raindrop_fall_velocity is None:
			self._stencil_raindrop_fall_velocity_initialize()
			
		# Extract the needed model variables
		self._in_rho[:,:,:] = state['air_density'].values[:,:,:,0]
		self._in_qr[:,:,:]  = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Extract the surface density
		rho_s = self._in_rho[:,:,-1:]
		self._in_rho_s[:,:,:] = np.repeat(rho_s, self._grid.nz, axis = 2)

		# Call the stencil's compute function
		self._stencil_raindrop_fall_velocity.compute()

		return self._out_vt


	def _stencil_adjustment_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of carrying out the saturation adjustment.
		"""
		# Initialize the GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_p   = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_ps  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_exn = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_T   = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencil outputs
		self._out_qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qc = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the stencil
		self._stencil_adjustment = gt.NGStencil(
			definitions_func = self._stencil_adjustment_defs,
			inputs = {'in_p': self._in_p, 'in_ps': self._in_ps, 'in_exn': self._in_exn, 
					  'in_T': self._in_T, 'in_qv': self._in_qv, 'in_qc': self._in_qc}, 
			global_inputs = {'dt': self._dt},
			outputs = {'out_qv': self._out_qv, 'out_qc': self._out_qc},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend)

	def _stencil_adjustment_defs(self, dt, in_p, in_ps, in_exn, in_T, in_qv, in_qc):
		"""
		GT4Py stencil carrying out the saturation adjustment.

		Parameters
		----------
		in_p : obj
			:class:`gridtools.Equation` representing the air pressure.
		in_ps : obj
			:class:`gridtools.Equation` representing the saturation vapor pressure.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_T : obj
			:class:`gridtools.Equation` representing the air temperature.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.

		Returns
		-------
		out_qv : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction of water vapor.
		out_qc : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction of cloud liquid water.

		References
		----------
		Doms, G., et al. (2015). *A description of the nonhydrostatic regional COSMO-model. \
			Part II: Physical parameterization.* Retrieved from `http://www.cosmo-model.org`_. \
		Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
			*Compute Unified Device Architecture (CUDA)-based parallelization of WRF Kessler \
			cloud microphysics scheme*. Computer \& Geosciences, 52:292-299.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the temporary fields
		tmp_qvs = gt.Equation()
		tmp_sat = gt.Equation()
		tmp_dlt = gt.Equation()

		# Instantiate the output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()

		# Compute the saturation mixing ratio of water vapor
		tmp_qvs[i, j, k] = self._beta * in_ps[i, j, k] / (in_p[i, j, k] - self._beta_c * in_ps[i, j, k])

		# Compute the amount of latent heat released by the condensation of cloud liquid water
		tmp_sat[i, j, k] = (tmp_qvs[i, j, k] - in_qv[i, j, k]) / \
						   (1. + self._kappa * in_ps[i, j, k] / 
							((in_T[i, j, k] - self._bw) * (in_T[i, j, k] - self._bw) * 
							 (in_p[i, j, k] - self._beta * in_ps[i, j, k]) * 
							 (in_p[i, j, k] - self._beta * in_ps[i, j, k])))

		# Compute the source term representing the evaporation of cloud liquid water
		tmp_dlt[i, j, k] = (tmp_sat[i, j, k] <= in_qc[i, j, k]) * tmp_sat[i, j, k] + \
						   (tmp_sat[i, j, k] > in_qc[i, j, k]) * in_qc[i, j, k]

		# Perform the adjustment
		out_qv[i, j, k] = in_qv[i, j, k] + tmp_dlt[i, j, k]
		out_qc[i, j, k] = in_qc[i, j, k] - tmp_dlt[i, j, k]

		return out_qv, out_qc

	def _stencil_raindrop_fall_velocity_initialize(self):
		"""
		Initialize the GT4Py stencil calculating the raindrop velocity.
		"""
		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_rho_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will serve as stencil output
		self._out_vt = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the stencil
		self._stencil_raindrop_fall_velocity = gt.NGStencil(
				definitions_func = self._stencil_raindrop_fall_velocity_defs,
				inputs = {'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s, 'in_qr': self._in_qr}, 
				outputs = {'out_vt': self._out_vt},
				domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
				mode = self._backend)

	def _stencil_raindrop_fall_velocity_defs(self, in_rho, in_rho_s, in_qr):
		"""
		GT4Py stencil calculating the raindrop velocity. 

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the density.
		in_rho_s : obj
			:class:`gridtools.Equation` representing the surface density.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the output field
		out_vt = gt.Equation()

		# Perform computations
		out_vt[i, j, k] = 36.34 * (1.e-3 * in_rho[i, j, k] * (in_qr[i, j, k] > 0.) * in_qr[i, j, k]) ** 0.1346 * \
						  ((in_rho_s[i, j, k] / in_rho[i, j, k]) ** 0.5)

		return out_vt
