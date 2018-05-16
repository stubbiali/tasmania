""" 
Two- and three-dimensional :math:`\mu`-terrain-following grids, with :math:`\mu` being
the SLEVE height-based vertical coordinate. 
"""
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import sys
import xarray as xr

import tasmania.namelist as nl 
from tasmania.grids.axis import Axis
from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.utils import equal_to as eq
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import smaller_or_equal_than as le
from tasmania.utils.utils import greater_than as gt

class SLEVE2d(GridXZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xz.GridXZ` to represent a rectangular and regular two-dimensional 
	grid embedded in a reference system whose coordinates are
		* the horizontal coordinate :math:`x`; 
		* the height-based SLEVE terrain-following coordinate :math:`\mu`.	
	The vertical coordinate :math:`\mu` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`\mu = \mu_F`, where 
	:math:`\mu`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	x : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`x`-axis.
	nx : int
		Number of grid points along :math:`x`.
	dx : float
		The :math:`x`-spacing.
	z : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`\mu`-main levels.
	z_on_interface_levels : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`\mu`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`\mu`-spacing.
	z_interface : float
		The interface coordinate :math:`\mu_F`.
	height : obj
		:class:`xarray.DataArray` representing the geometric height of the main levels.
	height_half_levels : obj
		:class:`xarray.DataArray` representing the geometric height of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\mu = \mu_F`.
	reference_pressure : obj
		:class:`xarray.DataArray` representing the reference pressure at the main levels.
	reference_pressure_half_levels : obj
		:class:`xarray.DataArray` representing the reference pressure at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_z, nz, units_x = 'm', dims_x = 'x', z_interface = None,
				 N = 100, s1 = 8.e3, s2 = 5.e3, topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{left}, ~ x_{right})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_z : tuple
			Tuple in the form :math:`(\mu_{top}, ~ \mu_{surface})`.
		nz : int
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`~tasmania.grids.axis.Axis.__init__`).
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`\mu_F`. If not specified, it is assumed that :math:`\mu_F = \mu_T`, with 
			:math:`\mu_T` the value of :math:`\mu` at the top of the domain. In other words, a fully terrain-following 
			coordinate system is supposed.
		N : `int`, optional
			Number of filter iterations performed to extract the large-scale component of the surface terrain-height. 
			Defaults to 100.
		s1 : `float`, optional
			Large-scale decay constant. Defaults to :math:`8000 ~ m`.
		s2 : `float`, optional 
			Small-scale decay constant. Defaults to :math:`5000 ~ m`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'. See :mod:`~tasmania.grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. 
			See :mod:`~tasmania.grids.topography` for further details.

		Keyword arguments
		-----------------
		**kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~tasmania.grids.topography.Topography1d`.
		"""
		# Preliminary checks
		if not (eq(domain_z[1], 0.) and gt(domain_z[0], 0.)):
			raise ValueError('SLEVE vertical coordinate should be positive' \
							 'and vanish at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_z, nz, units_x = units_x, dims_x = dims_x,
			units_z = '1', dims_z = 'atmosphere_hybrid_height_coordinate', # CF Conventions
			z_interface = z_interface, topo_type = topo_type, topo_time = topo_time, **kwargs)
		
		# Interface height
		self.height_interface = self.z_interface

		# Compute geometric height and refence pressure
		self._N, self._s1, self._s2 = N, s1, s2
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. In turn, the metric terms are re-computed.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elasped simulation time.
		"""
		self.topography.update(time)
		self._update_metric_terms()

	def plot(self, **kwargs):
		"""
		Plot the grid half levels using :mod:`matplotlib.pyplot`'s utilities.

		Keyword arguments
		----------
		**kwargs : 
			Keyword arguments to be forwarded to :func:`matplotlib.pyplot.subplots`.

		Note
		----
		For the sake of compliancy with the notation employed by `COSMO <http://www.cosmo-model.org>`_,
		the vertical geometric height is denoted by :math:`z`. 
		"""
		if not kwargs:
			kwargs = {'figsize': [11,8]}
		fig, ax = plt.subplots(**kwargs)

		# Shortcuts
		x, z_hl, zf = self.x.values, self.height_half_levels.values, self.height_interface
		
		for i in range(0, z_hl.shape[1]):
			ax.plot(x, z_hl[:,i], color = 'black')
		ax.fill_between(x, 0, z_hl[:,-1], color = 'gray')

		ax.set(xlabel = '$x$ [$m$]', ylabel = '$z$ [$m$]')

		if lt(zf, z_hl[0,0]): 
			ax.text(x[-1], zf, ' $z_F$', horizontalalignment = 'left',
					verticalalignment = 'center')
			ax.plot(np.array([x[0], x[-1]]), np.array([zf, zf]),
					color = 'black', linestyle = '--')

		ax.text(x[-1], z_hl[0,0], ' $z_T$', horizontalalignment = 'left',
				verticalalignment = 'center')

		plt.show()

	def _update_metric_terms(self):
		"""
		Update the class by computing the metric terms, i.e., the geometric height and the reference pressure, 
		at both half and main levels. In doing this, a logarithmic vertical profile of reference pressure is assumed. 
		This method should be called every time the topography is updated or changed.
		"""
		# Shortcuts
		hs = np.repeat(self.topography.topo.values[:,np.newaxis], self.nz+1, axis = 1)
		zv = np.reshape(self.z_on_interface_levels.values[:,np.newaxis], (1, self.nz+1))
		zt = zv[0,0]
		zf = self.z_interface
		N, s1, s2 = self._N, self._s1, self._s2
		
		# Apply low-high filter to surface-terrain height: a 3-points average operator is used
		h1 = np.copy(hs)
		for i in range(N):
			h1[1:-1,0] = 1./3. * (h1[:-2,0] + h1[1:-1,0] + h1[2:,0])
		h2 = hs - h1
		
		# Half levels geometric height
		a = np.repeat(zv, self.nx, axis = 0)
		b1 = np.sinh((zf - zv) / s1) / np.sinh(zf / s1) * lt(zv, zf)
		b1 = np.repeat(b1, self.nx, axis = 0)
		b2 = np.sinh((zf - zv) / s2) / np.sinh(zf / s2) * lt(zv, zf)
		b2 = np.repeat(b2, self.nx, axis = 0)
		z_hl = a + b1 * h1 + b2 * h2

		self.height_half_levels = xr.DataArray(z_hl, coords = [self.x.values, self.z_on_interface_levels.values],
											   dims = [self.x.dims, self.z_on_interface_levels.dims], 
											   attrs = {'units': 'm'})

		# Reference pressure at half levels
		if eq(nl.beta, 0.):
			p0_hl = nl.p_sl * np.exp(- nl.g * z_hl / (nl.Rd * nl.T_sl))
		else:
			p0_hl = nl.p_sl * np.exp(- nl.T_sl / nl.beta * \
				 	(1. - np.sqrt(1. - 2. * nl.beta * nl.g * z_hl / (nl.Rd * nl.T_sl**2))))

		self.reference_pressure_half_levels = xr.DataArray \
			(p0_hl, coords = [self.x.values, self.z_on_interface_levels.values],
			 dims = [self.x.dims, self.z_on_interface_levels.dims], attrs = {'units': 'Pa'})

		# Reference pressure at main levels
		self.reference_pressure = xr.DataArray(0.5 * (p0_hl[:,:-1] + p0_hl[:,1:]), 
											   coords = [self.x.values, self.z.values],
											   dims = [self.x.dims, self.z.dims], attrs = {'units': 'Pa'})

		# Main levels geometric height
		self.height = xr.DataArray(0.5 * (z_hl[:,:-1] + z_hl[:,1:]),
								   coords = [self.x.values, self.z.values],
								   dims = [self.x.dims, self.z.dims], attrs = {'units': 'm'})

class SLEVE3d(GridXYZ):
	"""
	This class inherits :class:`~tasmania.grids.grid_xyz.GridXYZ` to represent a rectangular and regular computational grid 
	embedded in a three-dimensional terrain-following reference system, whose coordinates are:
		* first horizontal coordinate :math:`x`, e.g., the longitude;
		* second horizontal coordinate :math:`y`, e.g., the latitude;
		* the SLEVE terrain-following coordinate :math:`\mu`.
	The vertical coordinate :math:`\mu` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`\mu = \mu_F`, where 
	:math:`\mu`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	xy_grid : obj
		:class:`~tasmania.grids.grid_xy.GridXY` representing the horizontal grid.
	z : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`z`-main levels.
	z_on_interface_levels : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`z`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`z`-spacing.
	z_interface : float
		The interface coordinate :math:`z_F`.
	topography : obj
		:class:`~tasmania.grids.topography.Topography2d` representing the underlying topography.
	height : obj
		:class:`xarray.DataArray` representing the geometric height of the main levels.
	height_half_levels : obj
		:class:`xarray.DataArray` representing the geometric height of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\mu = \mu_F`.
	reference_pressure : obj
		:class:`xarray.DataArray` representing the reference pressure at the main levels.
	reference_pressure_half_levels : obj
		:class:`xarray.DataArray` representing the reference pressure at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, domain_z, nz, 
				 units_x = 'degrees_east', dims_x = 'longitude', units_y = 'degrees_north', dims_y = 'latitude', 
				 z_interface = None, N = 100, s1 = 8.e3, s2 = 5.e3, 
				 topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{left}, ~ x_{right})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_y : tuple
			Tuple in the form :math:`(y_{left}, ~ y_{right})`.
		ny : int
			Number of grid points in the :math:`y`-direction.
		domain_z : tuple
			Tuple in the form :math:`(\mu_{top}, ~ \mu_{surface})`.
		nz : int
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`~tasmania.grids.axis.Axis.__init__`).
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`~tasmania.grids.axis.Axis.__init__`).
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`\mu_F`. If not specified, it is assumed that :math:`\mu_F = \mu_T`, with 
			:math:`\mu_T` the value of :math:`\mu` at the top of the domain. In other words, a fully terrain-following 
			coordinate nsystem is supposed.
		N : `int`, optional
			Number of filter iterations performed to determine the large-scale component of the surface terrain-height. 
			Defaults to 100.
		s1 : `float`, optional
			Large-scale decay constant. Defaults to :math:`8000 ~ m`.
		s2 : `float`, optional
			Small-scale decay constant. Defaults to :math:`5000 ~ m`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'. See :mod:`~tasmania.grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. 
			See :mod:`~tasmania.grids.topography` for further details.

		Keyword arguments
		-----------------
		**kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~tasmania.grids.topography.Topography2d`.
		"""
		# Preliminary checks
		if not (eq(domain_z[1], 0.) and gt(domain_z[0], 0.)):
			raise ValueError('SLEVE vertical coordinate should be positive' \
							 'and vanish at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_y, ny, domain_z, nz,
			units_x = units_x, dims_x = dims_x, units_y = units_y, dims_y = dims_y,
			units_z = '1', dims_z = 'atmosphere_hybrid_height_coordinate', # CF Conventions
			z_interface = z_interface, topo_type = topo_type, topo_time = topo_time, **kwargs)
		
		# Interface height
		self.height_interface = self.z_interface

		# Compute geometric height and refence pressure
		self._N, self._s1, self._s2 = N, s1, s2
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. In turn, the metric terms are re-computed.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self.topography.update(time)
		self._update_metric_terms()

	def _update_metric_terms(self):
		"""
		Update the class by computing the metric terms, i.e., the geometric height and the reference pressure, 
		at both half and main levels. In doing this, a logarithmic vertical profile of reference pressure is assumed. 
		This method should be called every time the topography is updated or changed.
		"""
		# Shortcuts
		hs = np.repeat(self.topography.topo.values[:,:,np.newaxis], self.nz+1, axis = 2)
		zv = np.reshape(self.z_on_interface_levels.values[:,np.newaxis,np.newaxis], (1, 1, self.nz+1))
		zt = zv[0,0]
		zf = self.z_interface
		N, s1, s2 = self._N, self._s1, self._s2

		# Apply low-high filter to surface-terrain height: a 9-points average operator is used
		h1 = np.copy(hs)
		for i in range(N):
			h1[1:-1,1:-1] = 1./9. * (h1[:-2,:-2] + h1[1:-1,:-2] + h1[2:,:-2] + \
									 h1[:-2,1:-1] + h1[1:-1,1:-1] + h1[2:,1:-1] + \
									 h1[:-2,2:] + h1[1:-1,2:] + h1[2:,2:])
		h2 = hs - h1

		# Half levels geometric height
		a = np.tile(zv, (self.nx, self.ny, 1))
		b1 = np.sinh((zf - zv) / s1) / np.sinh(zf / s1) * lt(zv, zf)
		b1 = np.tile(b1, (self.nx, self.ny, 1))
		b2 = np.sinh((zf - zv) / s2) / np.sinh(zf / s2) * lt(zv, zf)
		b2 = np.tile(b2, (self.nx, self.ny, 1))

		self.height_half_levels = xr.DataArray(z_hl,
			coords = [self.x.values, self.y.values, self.z_on_interface_levels.values],
			dims = [self.x.dims, self.y.dims, self.z_on_interface_levels.dims], attrs = {'units': 'm'})

		# Reference pressure at half levels
		if eq(nl.beta, 0.):
			p0_hl = nl.p_sl * np.exp(- nl.g * z_hl / (nl.Rd * nl.T_sl))
		else:
			p0_hl = nl.p_sl * np.exp(- nl.T_sl / nl.beta * \
					(1. - np.sqrt(1. - 2. * nl.beta * nl.g * z_hl / (nl.Rd * nl.T_sl**2))))

		self.reference_pressure_half_levels = xr.DataArray(p0_hl,
			coords = [self.x.values, self.y.values, self.z_on_interface_levels.values],
			dims = [self.x.dims, self.y.dims, self.z_on_interface_levels.dims], attrs = {'units': 'Pa'})

		# Reference pressure at main levels
		self.reference_pressure = xr.DataArray(0.5 * (p0_hl[:,:,:-1] + p0_hl[:,:,1:]), 
			coords = [self.x.values, self.y.values, self.z.values],
			dims = [self.x.dims, self.y.dims, self.z.dims], attrs = {'units': 'Pa'})

		# Main levels geometric height
		self.height = xr.DataArray(0.5 * (z_hl[:,:,:-1] + z_hl[:,:,1:]),
			coords = [self.x.values, self.y.values, self.z.values],
			dims = [self.x.dims, self.y.dims, self.z.dims], attrs = {'units': 'm'})

