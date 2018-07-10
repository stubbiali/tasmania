from datetime import timedelta
import numpy as np
import sympl

import tasmania.namelist as nl 
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.utils import equal_to as eq, smaller_than as lt, \
								 smaller_or_equal_than as le


class Sigma2d(GridXZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xz.GridXZ` to represent
	a rectangular and regular two-dimensional grid embedded in a reference system
	whose coordinates are

		* the horizontal coordinate :math:`x`; 
		* the pressure-based terrain-following coordinate \
			:math:`\sigma = p / p_{SL}`, where :math:`p` is the pressure and \
			:math:`p_{SL}` the pressure at the sea level.

	The vertical coordinate :math:`\sigma` may be formulated to define a hybrid
	terrain-following coordinate system with terrain-following coordinate lines
	between the surface terrain-height and :math:`\sigma = \sigma_F`, where
	:math:`\sigma`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	height : dataarray_like
		2-D :class:`sympl.DataArray` representing the geometric height
		of the main levels.
	height_on_interface_levels : dataarray_like
		2-D :class:`sympl.DataArray` representing the geometric height
		of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\sigma = \sigma_F`.
	reference_pressure : dataarray_like
		2-D :class:`sympl.DataArray` representing the reference pressure
		at the main levels.
	reference_pressure_on_interface_levels : dataarray_like
		2-D :class:`sympl.DataArray` representing the reference pressure
		at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_z, nz,
				 units_x='m', dims_x='x', z_interface=None, dtype=nl.datatype,
				 topo_type='flat_terrain', topo_time=timedelta(), topo_kwargs=None,
				 physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		domain_x : tuple
			The interval which the domain includes along the :math:`x`-axis.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_z : tuple
			The interval which the domain includes along the :math:`z`-axis.
			This should be specified in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate. Defaults to 'x'.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'm'.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that
			:math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the top
			of the domain. In other words, the coordinate system is supposed
			fully terrain-following.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'.
			See :class:`~tasmania.grids.topography.Topography1d` for further details.
		topo_time : `timedelta`, optional
			:class:`datetime.timedelta` representing the simulation time after
			which the topography should stop increasing. Default is 0, corresponding
			to a time-invariant terrain surface-height. See
			:mod:`~tasmania.grids.topography.Topography1d` for further details.
		topo_kwargs : `dict`, optional
			Keyword arguments to be forwarded to the constructor of
			:class:`~tasmania.grids.topography.Topography1d`.
		physical_constants : `dict`, optional
			Dictionary whose keys are the names of the physical constants used
			within this object, and whose values are the physical constants themselves.
			These are:

				* 'beta', the rate of increase in reference temperature with the \
					logarithm of reference pressure ([K ~ Pa:math:`^{-1}`]), \
					which defaults to :obj:`~tasmania.namelist.beta`;
				* 'g', the gravitational acceleration ([m s:math:`^{-2}`]), \
					which defaults to :obj:`~tasmania.namelist.g`;
				* 'p_sl', the reference pressure at sea level ([Pa]), \
					which defaults to :obj:`~tasmania.namelist.p_sl`;
				* 'Rd', the gas constant for dry air \
					([J K:math:`^{-1}` Kg:math:`^{-1}`]), \
					which defaults to :obj:`~tasmania.namelist.Rd`;
				* 'T_sl', the reference temperature at sea level ([K]), \
					which defaults to :obj:`~tasmania.namelist.T_sl`.

		Raises
		------
		ValueError :
			If the vertical coordinate is not positive, equal to one at the surface,
			or decreasing with height.
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# Preliminary checks
		if not (lt(domain_z[0], domain_z[1])
				and le(0., domain_z[0])
				and eq(domain_z[1], 1.)):
			raise ValueError('Pressure-based vertical coordinate should be positive, '
							 'equal to 1 at the surface, and decreasing with height.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_z, nz,
						 units_x=units_x, dims_x=dims_x,
						 units_z='1', dims_z='atmosphere_hybrid_sigma_pressure_coordinate',
						 z_interface=z_interface, dtype=dtype,
						 topo_type=topo_type, topo_time=topo_time,
						 topo_kwargs=topo_kwargs)

		# Keep track of the physical constants to use
		if physical_constants is None or not isinstance(physical_constants, dict):
			self._physical_constants = {}
		else:
			self._physical_constants = physical_constants

		# Get the physical constants to use
		beta = self._physical_constants.get('beta', nl.beta)
		g    = self._physical_constants.get('g', nl.g)
		Rd   = self._physical_constants.get('Rd', nl.Rd)
		T_sl = self._physical_constants.get('T_sl', nl.T_sl)

		# Interface height
		self.height_interface = Rd / g * np.log(1 / self.z_interface) * \
				  				(T_sl - 0.5 * beta * np.log(1 / self.z_interface))

		# Compute geometric height and reference pressure
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography, then re-compute the metric terms.

		Parameters
		----------
		time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		super().update_topography(time)
		self._update_metric_terms()

	def _update_metric_terms(self):
		"""
		Compute the metric terms, i.e., the geometric height and the
		reference pressure, at both half and main levels. In doing this,
		a logarithmic vertical profile of reference pressure is assumed.
		This method should be called every time the topography is updated or changed.
		"""
		# Extract the physical constants to use
		beta = self._physical_constants.get('beta', nl.beta)
		g    = self._physical_constants.get('g', nl.g)
		p_sl = self._physical_constants.get('p_sl', nl.p_sl)
		Rd   = self._physical_constants.get('Rd', nl.Rd)
		T_sl = self._physical_constants.get('T_sl', nl.T_sl)

		# Shortcuts
		hs = self.topography.topo.values
		zv = np.reshape(self.z_on_interface_levels.values[:, np.newaxis], (1, self.nz+1))
		zt = zv[0, 0]
		zf = self.z_interface
		
		# Reference pressure at the terrain surface
		if eq(beta, 0.):
			p0_s = p_sl * np.exp(- g * hs / (Rd*T_sl))
		else:
			p0_s = p_sl * np.exp(- T_sl / beta *
				   				 (1. - np.sqrt(1. - 2.*beta*g*hs / (Rd*T_sl*T_sl))))
		p0_s = np.repeat(p0_s[:, np.newaxis], self.nz+1, axis=1)

		# Reference pressure at the half levels
		a = p_sl * zv * (np.logical_and(le(zt, zv), le(zv, zf))) + \
			p_sl * zf * (1. - zv) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		a = np.repeat(a, self.nx, axis=0)
		b = (zv - zf) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		b = np.repeat(b, self.nx, axis=0)
		p0_hl = a + b * p0_s
		self.reference_pressure_on_interface_levels = \
			sympl.DataArray(p0_hl,
							coords=[self.x.values, self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.z_on_interface_levels.dims[0]],
							name='reference_pressure_on_interface_levels',
							attrs={'units': 'Pa'})

		# Geometric height at the half levels
		if eq(nl.beta, 0.):
			z_hl = nl.Rd * nl.T_sl / nl.g * np.log(nl.p_sl / p0_hl)
		else:
			z_hl = nl.Rd / nl.g * np.log(nl.p_sl / p0_hl) * \
				   (nl.T_sl - 0.5 * nl.beta * np.log(nl.p_sl / p0_hl))
		self.height_on_interface_levels = \
			sympl.DataArray(z_hl,
							coords=[self.x.values, self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.z_on_interface_levels.dims[0]],
							name='height_on_interface_levels',
							attrs={'units': 'm'})

		# Reference pressure at the main levels
		self.reference_pressure = sympl.DataArray(0.5 * (p0_hl[:, :-1] + p0_hl[:, 1:]),
												  coords=[self.x.values, self.z.values],
												  dims=[self.x.dims[0], self.z.dims[0]],
												  name='reference_pressure',
												  attrs={'units': 'Pa'})

		# Main levels geometric height
		self.height = sympl.DataArray(0.5 * (z_hl[:, :-1] + z_hl[:, 1:]),
									  coords=[self.x.values, self.z.values],
									  dims=[self.x.dims[0], self.z.dims[0]],
									  name='height',
									  attrs={'units': 'm'})


class Sigma3d(GridXYZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xyz.GridXYZ` to represent
	a rectangular and regular computational grid embedded in a three-dimensional
	terrain-following reference system, whose coordinates are:

		* the first horizontal coordinate :math:`x`, e.g., the longitude;
		* the second horizontal coordinate :math:`y`, e.g., the latitude;
		* the pressure-based terrain-following coordinate \
			:math:`\sigma = p / p_{SL}`, where :math:`p` is the pressure \
			and :math:`p_{SL}` the pressure at the sea level.

	The vertical coordinate :math:`\sigma` may be formulated to define a hybrid
	terrain-following coordinate system with terrain-following coordinate lines
	between the surface terrain-height and :math:`\sigma = \sigma_F`, where
	:math:`\sigma`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	height : dataarray_like
		3-D :class:`sympl.DataArray` representing the geometric height
		of the main levels.
	height_on_interface_levels : dataarray_like
		3-D :class:`sympl.DataArray` representing the geometric height
		of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\sigma = \sigma_F`.
	reference_pressure : dataarray_like
		3-D :class:`sympl.DataArray` representing the reference pressure
		at the main levels.
	reference_pressure_on_interface_levels : dataarray_like
		3-D :class:`sympl.DataArray` representing the reference pressure
		at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, domain_z, nz,
				 units_x='degrees_east', dims_x='longitude',
				 units_y='degrees_north', dims_y='latitude',
				 z_interface=None, dtype=nl.datatype,
				 topo_type='flat_terrain', topo_time=timedelta(), topo_kwargs=None,
				 physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		domain_x : tuple
			The interval which the domain includes along the :math:`x`-axis.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_y : tuple
			The interval which the domain includes along the :math:`y`-axis.
		ny : int
			Number of mass points along :math:`y`.
		domain_z : tuple
			The interval which the domain includes along the :math:`z`-axis.
			This should be specified in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate. Defaults to 'longitude'.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'degrees_east'.
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate. Defaults to 'latitude'.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'degrees_north'.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that
			:math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the
			top of the domain. In other words, the coordinate system is supposed
			fully terrain-following.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'.
			See :class:`~tasmania.grids.topography.Topography2d` for further details.
		topo_time : `timedelta`, optional
			:class:`datetime.timedelta` representing the simulation time after
			which the topography should stop increasing. Default is 0, corresponding
			to a time-invariant terrain surface-height.
			See :class:`~tasmania.grids.topography.Topography2d` for further details.
		topo_kwargs : `dict`, optional
			Keyword arguments to be forwarded to the constructor of
			:class:`~tasmania.grids.topography.Topography2d`.
		physical_constants : `dict`, optional
			Dictionary whose keys are the names of the physical constants used
			within this object, and whose values are the physical constants themselves.
			These are:

				* 'beta', the rate of increase in reference temperature with the \
					logarithm of reference pressure ([K ~ Pa:math:`^{-1}`]), \
					which defaults to :obj:`~tasmania.namelist.beta`;
				* 'g', the gravitational acceleration ([m s:math:`^{-2}`]), \
					which defaults to :obj:`~tasmania.namelist.g`;
				* 'p_sl', the reference pressure at sea level ([Pa]), \
					which defaults to :obj:`~tasmania.namelist.p_sl`;
				* 'Rd', the gas constant for dry air \
					([J K:math:`^{-1}` Kg:math:`^{-1}`]), \
					which defaults to :obj:`~tasmania.namelist.Rd`;
				* 'T_sl', the reference temperature at sea level ([K]), \
					which defaults to :obj:`~tasmania.namelist.T_sl`.

		Raises
		------
		ValueError :
			If the vertical coordinate is not positive, equal to one at the surface,
			or decreasing with height.
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# Preliminary checks
		if not (lt(domain_z[0], domain_z[1])
				and le(0., domain_z[0])
				and eq(domain_z[1], 1.)):
			raise ValueError('Pressure-based vertical coordinate should be positive, '
							 'equal to 1 at the surface, and decreasing with height.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_y, ny, domain_z, nz,
						 units_x=units_x, dims_x=dims_x,
						 units_y=units_y, dims_y=dims_y,
						 units_z='1', dims_z='atmosphere_hybrid_sigma_pressure_coordinate',
						 z_interface=z_interface, dtype=dtype,
						 topo_type=topo_type, topo_time=topo_time,
						 topo_kwargs=topo_kwargs)

		# Keep track of the physical constants to use
		if physical_constants is None or not isinstance(physical_constants, dict):
			self._physical_constants = {}
		else:
			self._physical_constants = physical_constants

		# Get the physical constants to use
		beta = self._physical_constants.get('beta', nl.beta)
		g    = self._physical_constants.get('g', nl.g)
		Rd   = self._physical_constants.get('Rd', nl.Rd)
		T_sl = self._physical_constants.get('T_sl', nl.T_sl)

		# Interface height
		self.height_interface = Rd / g * np.log(1. / self.z_interface) * \
								(T_sl - 0.5 * beta * np.log(1. / self.z_interface))

		# Compute geometric height and reference pressure
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography, then re-compute the metric terms.

		Parameters
		----------
		time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		super().update_topography(time)
		self._update_metric_terms()

	def _update_metric_terms(self):
		"""
		Compute the metric terms, i.e., the geometric height and the
		reference pressure, at both half and main levels. In doing this,
		a logarithmic vertical profile of reference pressure is assumed.
		This method should be called every time the topography is updated or changed.
		"""
		# Extract the physical constants to use
		beta = self._physical_constants.get('beta', nl.beta)
		g    = self._physical_constants.get('g', nl.g)
		p_sl = self._physical_constants.get('p_sl', nl.p_sl)
		Rd   = self._physical_constants.get('Rd', nl.Rd)
		T_sl = self._physical_constants.get('T_sl', nl.T_sl)

		# Shortcuts
		hs = np.repeat(self.topography_height[:, :, np.newaxis], self.nz+1, axis=2)
		zv = np.reshape(self.z_on_interface_levels.values[:, np.newaxis, np.newaxis],
						(1, 1, self.nz+1))
		zt = zv[0, 0]
		zf = self.z_interface

		# Reference pressure at the terrain surface
		if eq(beta, 0.):
			p0_s = p_sl * np.exp(- g * hs / (Rd * T_sl))
		else:
			p0_s = p_sl * np.exp(- T_sl / beta * \
				   (1. - np.sqrt(1. - 2.*beta*g*hs / (Rd*T_sl*T_sl))))

		# Reference pressure at the half levels
		a = p_sl * zv * (np.logical_and(le(zt, zv), le(zv, zf))) + \
			p_sl * zf * (1. - zv) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		a = np.tile(a, (self.nx, self.ny, 1))
		b = (zv - zf) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		b = np.tile(b, (self.nx, self.ny, 1))
		p0_hl = a + b * p0_s
		self.reference_pressure_on_interface_levels = \
			sympl.DataArray(p0_hl,
							coords=[self.x.values, self.y.values,
									self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.y.dims[0],
								  self.z_on_interface_levels.dims[0]],
							name='reference_pressure_on_interface_levels',
							attrs={'units': 'Pa'})

		# Geometric height at the half levels
		if eq(nl.beta, 0.):
			z_hl = nl.Rd * nl.T_sl / nl.g * np.log(nl.p_sl / p0_hl)
		else:
			z_hl = nl.Rd / nl.g * np.log(nl.p_sl / p0_hl) * \
				   (nl.T_sl - 0.5 * nl.beta * np.log(nl.p_sl / p0_hl))
		self.height_on_interface_levels = \
			sympl.DataArray(z_hl,
							coords=[self.x.values, self.y.values,
									self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.y.dims[0],
								  self.z_on_interface_levels.dims[0]],
							name='height_on_interface_levels',
							attrs={'units': 'm'})

		# Reference pressure at the main levels
		self.reference_pressure = \
			sympl.DataArray(0.5 * (p0_hl[:, :, :-1] + p0_hl[:, :, 1:]),
							coords=[self.x.values, self.y.values, self.z.values],
							dims=[self.x.dims[0], self.y.dims[0], self.z.dims[0]],
							name='reference_pressure', attrs={'units': 'Pa'})

		# Geometric height at the main levels
		self.height = \
			sympl.DataArray(0.5 * (z_hl[:, :, :-1] + z_hl[:, :, 1:]),
							coords=[self.x.values, self.y.values, self.z.values],
							dims=[self.x.dims[0], self.y.dims[0], self.z.dims[0]],
							name='height', attrs={'units': 'm'})
