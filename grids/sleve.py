from datetime import timedelta
import numpy as np
import sympl

import tasmania.namelist as nl 
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.utils import equal_to as eq, smaller_than as lt, \
								 greater_than as gt


class SLEVE2d(GridXZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xz.GridXZ` to represent
	a rectangular and regular grid embedded in a two-dimensional reference
	system whose coordinates are

		* the horizontal coordinate :math:`x`; 
		* the height-based SLEVE terrain-following coordinate :math:`\mu`.

	The vertical coordinate :math:`\mu` may be formulated to define a hybrid
	terrain-following coordinate system with terrain-following coordinate lines
	between the surface terrain-height and :math:`\mu = \mu_F`, where
	:math:`\mu`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	height : dataarray_like
		2-D :class:`sympl.DataArray` representing the geometric height
		of the main levels.
	height_on_interface_levels : dataarray_like
		2-D :class:`sympl.DataArray` representing the geometric height
		of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\mu = \mu_F`.
	reference_pressure : dataarray_like
		2-D :class:`sympl.DataArray` representing the reference pressure
		at the main levels.
	reference_pressure_on_interface_levels : dataarray_like
		2-D :class:`sympl.DataArray` representing the reference pressure
		at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_z, nz,
				 units_x='m', dims_x='x',
				 z_interface=None, dtype=nl.datatype,
				 topo_type='flat_terrain', topo_time=timedelta(), topo_kwargs=None,
				 physical_constants=None, niter=100, s1=8.e3, s2=5.e3):
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

		niter : `int`, optional
			Number of filter iterations performed to extract the large-scale
			component of the surface terrain-height. Defaults to 100.
		s1 : `float`, optional
			Large-scale decay constant. Defaults to :math:`8000` m.
		s2 : `float`, optional
			Small-scale decay constant. Defaults to :math:`5000` m.

		Raises
		------
		ValueError :
			If the vertical coordinate either assumes negative values, or
			does not vanish at the terrain surface.
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# Preliminary checks
		if not (eq(domain_z[1], 0.) and gt(domain_z[0], 0.)):
			raise ValueError('SLEVE vertical coordinate should be positive '
							 'and vanish at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_z, nz,
						 units_x=units_x, dims_x=dims_x,
						 units_z='1', dims_z='atmosphere_hybrid_height_coordinate',
						 z_interface=z_interface, dtype=dtype,
						 topo_type=topo_type, topo_time=topo_time,
						 topo_kwargs=topo_kwargs)

		# Interface height
		self.height_interface = self.z_interface

		# Keep track of input arguments
		self._niter, self._s1, self._s2 = niter, s1, s2
		if physical_constants is None or not isinstance(physical_constants, dict):
			self._physical_constants = {}
		else:
			self._physical_constants = physical_constants

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
		hs = np.repeat(self.topography.topo.values[:, np.newaxis], self.nz+1, axis=1)
		zv = np.reshape(self.z_on_interface_levels.values[:, np.newaxis], (1, self.nz+1))
		zf = self.z_interface
		n, s1, s2 = self._niter, self._s1, self._s2
		
		# Apply low-high filter to surface-terrain height
		# A 3-points average operator is used
		h1 = np.copy(hs)
		for _ in range(n):
			h1[1:-1, 0] = 1./3. * (h1[:-2, 0] + h1[1:-1, 0] + h1[2:, 0])
		h2 = hs - h1
		
		# Geometric height at the interface levels
		a = np.repeat(zv, self.nx, axis=0)
		b1 = np.sinh((zf - zv) / s1) / np.sinh(zf / s1) * lt(zv, zf)
		b1 = np.repeat(b1, self.nx, axis=0)
		b2 = np.sinh((zf - zv) / s2) / np.sinh(zf / s2) * lt(zv, zf)
		b2 = np.repeat(b2, self.nx, axis=0)
		z_hl = a + b1 * h1 + b2 * h2
		self.height_on_interface_levels = \
			sympl.DataArray(z_hl,
							coords=[self.x.values, self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.z_on_interface_levels.dims[0]],
							name='height_on_interface_levels',
							attrs={'units': 'm'})

		# Reference pressure at the interface levels
		if eq(beta, 0.):
			p0_hl = p_sl * np.exp(- g * z_hl / (Rd*T_sl))
		else:
			p0_hl = p_sl * np.exp(- T_sl / beta *
				 	(1. - np.sqrt(1. - 2.*beta*g*z_hl / (Rd*T_sl**2))))
		self.reference_pressure_on_interface_levels = \
			sympl.DataArray(p0_hl,
							coords=[self.x.values, self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.z_on_interface_levels.dims[0]],
							name='reference_pressure_on_interface_levels',
							attrs={'units': 'Pa'})

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


class SLEVE3d(GridXYZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xyz.GridXYZ` to represent
	a rectangular and regular computational grid embedded in a three-dimensional
	terrain-following reference system, whose coordinates are:

		* the first horizontal coordinate :math:`x`, e.g., the longitude;
		* the second horizontal coordinate :math:`y`, e.g., the latitude;
		* the SLEVE terrain-following coordinate :math:`\mu`.

	The vertical coordinate :math:`\mu` may be formulated to define a hybrid
	terrain-following coordinate system with terrain-following coordinate lines
	between the surface terrain-height and :math:`\mu = \mu_F`, where
	:math:`\mu`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	height : dataarray_like
		3-D :class:`sympl.DataArray` representing the geometric height
		of the main levels.
	height_on_interface_levels : dataarray_like
		3-D :class:`sympl.DataArray` representing the geometric height
		of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\mu = \mu_F`.
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
				 physical_constants=None, niter=100, s1=8.e3, s2=5.e3):
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

		niter : `int`, optional
			Number of filter iterations performed to extract the large-scale
			component of the surface terrain-height. Defaults to 100.
		s1 : `float`, optional
			Large-scale decay constant. Defaults to :math:`8000` m.
		s2 : `float`, optional
			Small-scale decay constant. Defaults to :math:`5000` m.

		Raises
		------
		ValueError :
			If the vertical coordinate either assumes negative values, or
			does not vanish at the terrain surface.
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# Preliminary checks
		if not (eq(domain_z[1], 0.) and gt(domain_z[0], 0.)):
			raise ValueError('SLEVE vertical coordinate should be positive '
							 'and vanish at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_y, ny, domain_z, nz,
						 units_x=units_x, dims_x=dims_x,
						 units_y=units_y, dims_y=dims_y,
						 units_z='1', dims_z='atmosphere_hybrid_height_coordinate',
						 z_interface=z_interface, dtype=dtype,
						 topo_type=topo_type, topo_time=topo_time,
						 topo_kwargs=topo_kwargs)

		# Interface height
		self.height_interface = self.z_interface

		# Keep track of the input arguments
		self._niter, self._s1, self._s2 = niter, s1, s2
		if physical_constants is None or not isinstance(physical_constants, dict):
			self._physical_constants = {}
		else:
			self._physical_constants = physical_constants

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
		hs = np.repeat(self.topography.topo.values[:, :, np.newaxis], self.nz+1, axis=2)
		zv = np.reshape(self.z_on_interface_levels.values[:, np.newaxis, np.newaxis],
						(1, 1, self.nz+1))
		zf = self.z_interface
		n, s1, s2 = self._niter, self._s1, self._s2

		# Apply low-high filter to surface-terrain height
		# A 9-points average operator is used
		h1 = np.copy(hs)
		for _ in range(n):
			h1[1:-1, 1:-1] = 1./9. * (h1[:-2, :-2] + h1[1:-1, :-2] + h1[2:, :-2] +
									  h1[:-2, 1:-1] + h1[1:-1, 1:-1] + h1[2:, 1:-1] +
									  h1[:-2, 2:] + h1[1:-1, 2:] + h1[2:, 2:])
		h2 = hs - h1

		# Geometric height at the interface levels
		a = np.tile(zv, (self.nx, self.ny, 1))
		b1 = np.sinh((zf - zv) / s1) / np.sinh(zf / s1) * lt(zv, zf)
		b1 = np.tile(b1, (self.nx, self.ny, 1))
		b2 = np.sinh((zf - zv) / s2) / np.sinh(zf / s2) * lt(zv, zf)
		b2 = np.tile(b2, (self.nx, self.ny, 1))
		z_hl = a + b1 * h1 + b2 * h2
		self.height_on_interface_levels = \
			sympl.DataArray(z_hl,
							coords=[self.x.values, self.y.values,
									self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.y.dims[0],
								  self.z_on_interface_levels.dims[0]],
							name='height_on_interface_levels',
							attrs={'units': 'm'})

		# Reference pressure at the interface levels
		if eq(beta, 0.):
			p0_hl = p_sl * np.exp(- g * z_hl / (Rd*T_sl))
		else:
			p0_hl = p_sl * np.exp(- T_sl / beta *
								  (1. - np.sqrt(1. - 2.*beta*g*z_hl / (Rd*T_sl**2))))
		self.reference_pressure_on_interface_levels = \
			sympl.DataArray(p0_hl,
							coords=[self.x.values, self.y.values,
									self.z_on_interface_levels.values],
							dims=[self.x.dims[0], self.y.dims[0],
								  self.z_on_interface_levels.dims[0]],
							name='reference_pressure_on_interface_levels',
							attrs={'units': 'Pa'})

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
