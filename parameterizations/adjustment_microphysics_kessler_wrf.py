import abc
import numpy as np

import gridtools as gt
from tasmania.namelist import cp, datatype, L, Rd, Rv
from tasmania.parameterizations.adjustments import AdjustmentMicrophysics
from tasmania.storages.grid_data import GridData
import tasmania.utils.utils as utils
import tasmania.utils.utils_meteo as utils_meteo

class AdjustmentMicrophysicsKesslerWRF(AdjustmentMicrophysics):
	"""
	This class inherits :class:`~parameterizations.adjustment_microphysics.AdjustmentMicrophysics` 
	to implement the WRF version of the Kessler scheme.

	Attributes
	----------
	a : float
		Autoconversion threshold, in units of [:math:`g ~ kg^{-1}`].
	k1 : float
		Rate of autoconversion, in units of [:math:`s^{-1}`].
	k2 : float
		Rate of autoaccretion, in units of [:math:`s^{-1}`].
	"""
	def __init__(self, grid, rain_evaporation_on, backend, a = .5, k1 = .001, k2 = 2.2):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.

		Keyword arguments
		-----------------
		a : float
			Autoconversion threshold, in units of [:math:`g ~ kg^{-1}`]. Default is :math:`0.5 ~ g ~ kg^{-1}`.
		k1 : float
			Rate of autoconversion, in units of [:math:`s^{-1}`]. Default is :math:`0.001 ~ s^{-1}`.
		k2 : float
			Rate of autoaccretion, in units of [:math:`s^{-1}`]. Default is :math:`2.2 ~ s^{-1}`.

		Note
		----
		To instantiate this class, one should prefer the static method 
		:meth:`~parameterizations.adjustment_microphysics.AdjustmentMicrophysics.factory` of 
		:class:`~parameterizations.adjustment_microphysics.AdjustmentMicrophysics`.
		"""
		super().__init__(grid, rain_evaporation_on, backend)
		self.a, self.k1, self.k2 = a, k1, k2

		# Constants for Tetens formula
		self._p0 = 610.78
		self._alpha = 17.27
		self._Tr = 273.15
		self._bw = 35.85
		
		# Shortcuts
		self._beta   = Rd / Rv
		self._beta_c = 1. - self._beta
		self._kappa  = L * self._alpha * self._beta * (self._Tr - self._bw) / cp 

		# Initialize pointers to the underlying GT4Py's stencils
		# They will be properly re-directed the first time the entry point method is invoked
		self._stencil_auxiliary = None
		self._stencil_adjustment = None
		self._stencil_raindrop_fall_velocity = None

	def __call__(self, dt, state):
		"""
		Entry-point method performing microphysics adjustments and computing microphysics diagnostics.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the timestep.
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* air_temperature (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		Returns
		-------
		state_new : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the adjusted state.
			It contains the following updated variables:

			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		diagnostics : obj
			:class:`~storages.grid_data.GridData` storing the following fields:
			
			* change_over_time_in_air_potential_temperature (unstaggered).
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_auxiliary is None:
			self._stencils_initialize()

		# Initialize the output state
		time = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		state_new = type(state)(time, self._grid)

		# Update the local time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Extract the required model variables
		self._in_rho[:,:,:] = state['air_density'].values[:,:,:,0]
		self._in_p[:,:,:]   = state['air_pressure'].values[:,:,:,0]
		self._in_exn[:,:,:] = state['exner_function'].values[:,:,:,0]
		self._in_T[:,:,:]	= state['air_temperature'].values[:,:,:,0]
		self._in_qv[:,:,:]  = state['mass_fraction_of_water_vapor_in_air'].values[:,:,:,0]
		self._in_qc[:,:,:]  = state['mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0]
		self._in_qr[:,:,:]  = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Compute the saturation water vapor pressure via Tetens' formula
		self._in_ps[:,:,:] = self._p0 * np.exp(self._alpha * (self._in_T[:,:,:] - self._Tr) / (self._in_T[:,:,:] - self._bw))

		# Run the auxiliary stencil
		self._stencil_auxiliary.compute()

		# Compute the source terms representing accretion and evaporation
		self._in_Cr[:,:,:] = self._aux_Cr[:,:,:] * (self._in_qr[:,:,:] ** .875)
		if self._rain_evaporation_on:
			C = 1.6 + 124.9 * (self._aux_Er_3 ** .2046)
			self._in_Er[:,:,:] = self._aux_Er_1[:,:,:] * self._aux_Er_2[:,:,:] * C[:,:,:] * (self._aux_Er_3[:,:,:]) ** .525 / \
								 (5.4e5 + 2.55e6 / self._aux_Er_4[:,:,:])

		# Run the adjustment stencil
		self._stencil_adjustment.compute()

		# Update the output state
		state_new.add(mass_fraction_of_water_vapor_in_air         = self._out_qv,
					  mass_fraction_of_cloud_liquid_water_in_air  = self._out_qc,
					  mass_fraction_of_precipitation_water_in_air = self._out_qr)

		# Collect the diagnostics
		time = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		diagnostics = GridData(time, self._grid, change_over_time_in_air_potential_temperature = self._out_w)

		return state_new, diagnostics

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
		# If this is the first time this method is invoked, initialize the auxiliary GT4Py's stencil
		if self._stencil_raindrop_fall_velocity is None:
			self._stencil_raindrop_fall_velocity_initialize()
			
		# Extract the needed model variables
		self._in_rho[:,:,:] = state['air_density'].values[:,:,:,0]
		self._in_qr[:,:,:]  = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Update the Numpy arrays which serve as inputs to the stencil
		self._stencil_raindrop_fall_velocity_set_inputs(rho, qr)

		# Call the stencil's compute function
		self._stencil_raindrop_fall_velocity.compute()

		# Compute the raindrop fall velocity
		vt = 36.34 * ((self._out_prod) ** .1346) * np.sqrt(self._out_div)

		return vt


	def _stencils_initialize(self):
		"""
		Initialize the GT4Py's stencils.
		"""
		# Initialize the GT4Py's Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_p   = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_ps  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_exn = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_T   = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Cr  = np.zeros((nx, ny, nz), dtype = datatype)
		if self._rain_evaporation_on:
			self._in_Er = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencil outputs
		self._tmp_qvs  = np.zeros((nx, ny, nz), dtype = datatype)
		self._tmp_Ar   = np.zeros((nx, ny, nz), dtype = datatype)
		self._aux_Cr   = np.zeros((nx, ny, nz), dtype = datatype)
		self._aux_Er_1 = np.zeros((nx, ny, nz), dtype = datatype)
		self._aux_Er_2 = np.zeros((nx, ny, nz), dtype = datatype)
		self._aux_Er_3 = np.zeros((nx, ny, nz), dtype = datatype)
		self._aux_Er_4 = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qv   = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qc   = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qr   = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_w    = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_sat  = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the auxiliary stencil
		self._stencil_auxiliary = gt.NGStencil(
			definitions_func = self._stencil_auxiliary_defs,
			inputs = {'in_rho': self._in_rho, 'in_p': self._in_p, 'in_ps': self._in_ps, 
					  'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr}, 
			outputs = {'out_qvs': self._tmp_qvs, 'out_Ar': self._tmp_Ar, 'aux_Cr': self._aux_Cr, 
					   'aux_Er_1': self._aux_Er_1, 'aux_Er_2': self._aux_Er_2, 
					   'aux_Er_3': self._aux_Er_3, 'aux_Er_4': self._aux_Er_4},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend)

		# Initialize the adjustment stencil
		_inputs  = {'in_p': self._in_p, 'in_ps': self._in_ps, 'in_exn': self._in_exn, 'in_T': self._in_T, 
					'in_qv': self._in_qv, 'in_qvs': self._tmp_qvs, 'in_qc': self._in_qc, 'in_qr': self._in_qr, 
					'in_Ar': self._tmp_Ar, 'in_Cr': self._in_Cr}
		_outputs = {'tmp_sat': self._out_sat, 'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr, 
					'out_w': self._out_w}
		if self._rain_evaporation_on:
			_inputs['in_Er'] = self._in_Er
		
		self._stencil_adjustment = gt.NGStencil(
					definitions_func = self._stencil_adjustment_defs,
					inputs = _inputs,
					global_inputs = {'dt': self._dt},
					outputs = _outputs,
					domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
					mode = self._backend)

	def _stencil_auxiliary_defs(self, in_rho, in_p, in_ps, in_qv, in_qc, in_qr):
		"""
		GT4Py's stencil computing auxiliary quantities required by the Kessler scheme.

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_p : obj
			:class:`gridtools.Equation` representing the air pressure.
		in_ps : obj
			:class:`gridtools.Equation` representing the saturation vapor pressure.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		out_qvs : obj
			:class:`gridtools.Equation` representing the saturation mass fraction of water vapor.
		tmp_Ar : obj
			:class:`gridtools.Equation` representing the source term accounting for the contribution of autoconversion to
			the development of rain.
		aux_Cr : obj
			:class:`gridtools.Equation` representing the product between the rate of accretion and the mass fraction of 
			cloud liquid water. 
			This is required to compute the source term accounting for the contribution of accretion to the growth of rain.
		aux_Er_1 : `obj`, optional
			:class:`gridtools.Equation` representing the inverse of air density, expressed in [:math:`g ~ cm^{-3}`]. 
			As this is required to compute the source term accounting for the evaporation of rain, it is calculated only if
			evaporation is swithed on.
		aux_Er_2 : `obj`, optional
			:class:`gridtools.Equation` representing the difference between the unity and the ratio between the mass fraction 
			of water vapor and the saturation mass fraction of water vapor. 
			As this is required to compute the source term accounting for the evaporation of rain, it is calculated only if
			evaporation is swithed on.
		aux_Er_3 : `obj`, optional
			:class:`gridtools.Equation` representing the product between air density, expressed in [:math:`g ~ cm^{-3}`], 
			and the mass fraction of precipitation water.
			As this is required to compute the source term accounting for the evaporation of rain, it is calculated only if
			evaporation is swithed on.
		aux_Er_4 : `obj`, optional
			:class:`gridtools.Equation` representing the product between the pressure, expressed in [:math:`mbar`], and the
			saturation mass fraction of water vapor.
			As this is required to compute the source term accounting for the evaporation of rain, it is calculated only if
			evaporation is swithed on.

		References
		----------
		Doms, G., et al. (2015). *A description of the nonhydrostatic regional COSMO-model. Part II: Physical parameterization.* \
			Retrieved from `http://www.cosmo-model.org`_. \
		Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). *Compute Unified Device Architecture \
			(CUDA)-based parallelization of WRF Kessler cloud microphysics scheme*. Computer \& Geosciences, 52:292-299.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the temporary fields
		tmp_p	     = gt.Equation()
		tmp_p_mbar   = gt.Equation()
		tmp_rho_gcm3 = gt.Equation()

		# Instantiate the output fields
		out_qvs  = gt.Equation()
		out_Ar   = gt.Equation()
		aux_Cr   = gt.Equation()
		aux_Er_1 = gt.Equation()
		aux_Er_2 = gt.Equation()
		aux_Er_3 = gt.Equation()
		aux_Er_4 = gt.Equation()

		# Compute pressure at model main levels
		tmp_p[i, j, k] = .5 * (in_p[i, j, k] + in_p[i, j, k+1])

		# Perform units conversion
		tmp_rho_gcm3[i, j, k] = 1.e3 * in_rho[i, j, k]
		tmp_p_mbar[i, j, k] = 1.e-2 * tmp_p[i, j, k]

		# Compute the saturation mixing ratio of water vapor
		out_qvs[i, j, k] = self._beta * in_ps[i, j, k] / (tmp_p[i, j, k] - self._beta_c * in_ps[i, j, k])

		# Compute the autoconversion
		out_Ar[i, j, k] = self.k1 * (in_qc[i, j, k] > self.a) * (in_qc[i, j, k] - self.a)

		# Compute the auxiliary terms for accretion
		aux_Cr[i, j, k] = self.k2 * in_qc[i, j, k]

		if self._rain_evaporation_on:
			# Compute the auxiliary terms for evaporation
			aux_Er_1[i, j, k] = 1. / tmp_rho_gcm3[i, j, k]
			aux_Er_2[i, j, k] = 1. - in_qv[i, j, k] / out_qvs[i, j, k]
			aux_Er_3[i, j, k] = tmp_rho_gcm3[i, j, k] * in_qr[i, j, k]
			aux_Er_4[i, j, k] = tmp_p_mbar[i, j, k] * out_qvs[i, j, k]

			return out_qvs, out_Ar, aux_Cr, aux_Er_1, aux_Er_2, aux_Er_3, aux_Er_4

		return out_qvs, out_Ar, aux_Cr

	def _stencil_adjustment_defs(self, dt, in_p, in_ps, in_exn, in_T, in_qv, in_qvs, in_qc, in_qr, in_Ar, in_Cr, in_Er = None):
		"""
		GT4Py's stencil carrying out the microphysical adjustments and computing the change over time in potential temperature.

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
		in_qvs : obj
			:class:`gridtools.Equation` representing the saturaion mass fraction of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.
		in_Ar : obj
			:class:`gridtools.Equation` representing the source term accounting for the contribution of autoconversion
			to the development of rain.
		in_Cr : obj
			:class:`gridtools.Equation` representing the source term accounting for the contribution of accretion
			to the development of rain.
		in_Er : `obj`, optional
			:class:`gridtools.Equation` representing the source term accounting for the evaporation of rain.

		Returns
		-------
		out_qv : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction of water vapor.
		out_qc : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction of cloud liquid water.
		out_qr : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction of precipitation water.
		out_w : obj
			:class:`gridtools.Equation` representing the change over time in potential temperature.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the temporary fields
		tmp_p   = gt.Equation()
		tmp_qc  = gt.Equation()
		tmp_sat = gt.Equation()
		tmp_dlt = gt.Equation()
		if self._rain_evaporation_on:
			tmp_qv = gt.Equation()

		# Instantiate the output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()
		out_qr = gt.Equation()
		out_w  = gt.Equation()

		# Compute pressure at model main levels
		tmp_p[i, j, k] = .5 * (in_p[i, j, k] + in_p[i, j, k+1])

		# Perform the adjustments, neglecting the evaporation of cloud liquid water
		if not self._rain_evaporation_on:
			tmp_qc[i, j, k] = in_qc[i, j, k] - self.time_levels * dt * (in_Ar[i, j, k] + in_Cr[i, j, k])
			out_qr[i, j, k] = in_qr[i, j, k] + self.time_levels * dt * (in_Ar[i, j, k] + in_Cr[i, j, k])
		else:
			tmp_qv[i, j, k] = in_qv[i, j, k] + self.time_levels * dt * in_Er[i, j, k]
			tmp_qc[i, j, k] = in_qc[i, j, k] - self.time_levels * dt * (in_Ar[i, j, k] + in_Cr[i, j, k])
			out_qr[i, j, k] = in_qr[i, j, k] + self.time_levels * dt * (in_Ar[i, j, k] + in_Cr[i, j, k] - in_Er[i, j, k])

		# Compute the source term representing the evaporation of cloud liquid water
		if not self._rain_evaporation_on:
			tmp_sat[i, j, k] = (in_qvs[i, j, k] - in_qv[i, j, k]) / \
							   (1. + self._kappa * in_ps[i, j, k] / 
							    ((in_T[i, j, k] - self._bw) * (in_T[i, j, k] - self._bw) * 
							   	 (in_p[i, j, k] - self._beta * in_ps[i, j, k]) * (in_p[i, j, k] - self._beta * in_ps[i, j, k])))
		else:
			tmp_sat[i, j, k] = (in_qvs[i, j, k] - tmp_qv[i, j, k]) / \
							   (1. + self._kappa * in_ps[i, j, k] / 
							    ((in_T[i, j, k] - self._bw) * (in_T[i, j, k] - self._bw) * 
							   	 (in_p[i, j, k] - self._beta * in_ps[i, j, k]) * (in_p[i, j, k] - self._beta * in_ps[i, j, k])))

		tmp_dlt[i, j, k] = (tmp_sat[i, j, k] <= tmp_qc[i, j, k]) * tmp_sat[i, j, k] + \
						   (tmp_sat[i, j, k] > tmp_qc[i, j, k]) * tmp_qc[i, j, k]

		# Perform the adjustments, accounting for the evaporation of cloud liquid water
		if not self._rain_evaporation_on:
			out_qv[i, j, k] = in_qv[i, j, k] + tmp_dlt[i, j, k]
			out_qc[i, j, k] = tmp_qc[i, j, k] - tmp_dlt[i, j, k]
		else:
			out_qv[i, j, k] = tmp_qv[i, j, k] + tmp_dlt[i, j, k]
			out_qc[i, j, k] = tmp_qc[i, j, k] - tmp_dlt[i, j, k]

		# Compute the change over time in potential temperature
		if not self._rain_evaporation_on:
			out_w[i, j, k] = (- L) / (.5 * (in_exn[i, j, k] + in_exn[i, j, k+1])) * tmp_dlt[i, j, k]
		else:
			out_w[i, j, k] = (- L) / (.5 * (in_exn[i, j, k] + in_exn[i, j, k+1])) * (tmp_dlt[i, j, k] + in_Er[i, j, k])

		return tmp_sat, out_qv, out_qc, out_qr, out_w


	def _stencil_raindrop_fall_velocity_initialize(self):
		"""
		Initialize the GT4Py's stencil providing auxiliary quantitites required to compute the raindrop velocity.
		"""
		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho   = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_rho_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qr    = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencil outputs
		self._out_prod = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_div  = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the stencil
		self._stencil_raindrop_fall_velocity = gt.NGStencil(
				definitions_func = self._stencil_raindrop_fall_velocity_defs,
				inputs = {'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s, 'in_qr': self._in_qr}, 
				outputs = {'out_prod': self._out_prod, 'out_div': self._out_div},
				domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
				mode = self._backend)

	def _stencil_raindrop_fall_velocity_defs(self, in_rho, in_rho_s, in_qr):
		"""
		GT4Py's stencil providing auxiliary quantitites required to compute the raindrop velocity. 

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
		out_prod : obj
			:class:`gridtools.Equation` representing the product between air density and mass fraction of precipitation water,
			in units of [:math:`g ~ cm^{-3}`].
		out_prod : obj
			:class:`gridtools.Equation` representing the ration between surface air density and air density.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the output fiels
		out_prod = gt.Equation()
		out_div  = gt.Equation()

		# Perform computations
		out_prod[i, j, k] = 1.e3 * in_rho[i, j, k] * in_qr[i, j, k]
		out_div[i, j, k]  = in_rho_s[i, j, k] / in_rho[i, j, k]

		return out_prod, out_div
