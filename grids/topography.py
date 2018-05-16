""" 
Classes representing one- and two-dimensional topographies, possibly time-dependent. 
Indeed, although clearly not physical, a terrain surface (slowly) growing in the early stages
of a simulation may help to retrieve numerical stability, as it prevents steep gradients
in the first few iterations.

Letting :math:`h_s = h_s(x)` be a one-dimensional topography, with :math:`x \in [a,b]`,
the user may choose among:
	* a flat terrain, i.e., :math:`h_s(x) \equiv 0`;
	* a Gaussian-shaped mountain, i.e., 
	
		.. math::
			h_s(x) = h_{max} \exp{\left[ - \left( \\frac{x - c}{\sigma_x} \\right)^2 \\right]}, 
			
	  where :math:`c = 0.5 (a + b)`.

For the two-dimensional case, letting :math:`h_s = h_s(x,y)` be the topography, with 
:math:`x \in [a_x,b_x]` and :math:`y \in [a_y,b_y]`, the following profiles are provided:
	* flat terrain, i.e., :math:`h_s(x,y) \equiv 0`;
	* Gaussian shaped-mountain, i.e. 

		.. math:: 
			h_s(x,y) = h_{max} \exp{\left[ - \left( \\frac{x - c_x}{\sigma_x} \\right)^2 - \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]} ;
			
	* modified Gaussian-shaped mountain proposed by Schaer and Durran (1997), 

		.. math::
			h_s(x,y) = \\frac{h_{max}}{\left[ 1 + \left( \\frac{x - c_x}{\sigma_x} \\right)^2 + \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]^{3/2}}.

Yet, user-defined profiles are supported as well, provided that they admit an analytical expression. 
This is passed to the class as a string, which is then parsed in C++ via `Cython <http://cython.org>`_ 
(see :class:`~grids.parser.parser_1d` and :class:`~grids.parser.parser_2d`). Hence, the string itself must be 
fully C++-compliant.

References:
	Schaer, C., and Durran, D. R. (1997). *Vortex formation and vortex shedding in continuosly stratified flows \
	past isolated topography*. Journal of Atmospheric Sciences, 54:534-554.
"""
from datetime import timedelta
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xarray as xr

from tasmania.namelist import datatype
from tasmania.utils.utils import equal_to as eq
from tasmania.utils.utils import smaller_than as lt

class Topography1d:
	"""
	Class representing a one-dimensional topography.
	
	Attributes
	----------
	topo : array_like
		:class:`xarray.DataArray` representing the topography (in meters).
	topo_type : str
		Topography type. Either:

			* 'flat_terrain'; 
			* 'gaussian';
			* 'user_defined'.

	topo_time : obj
		:class:`datetime.timedelta` object representing the elapsed simulation time after which the topography 
		should stop increasing.
	topo_fact : float
		Topography factor. It runs in between 0 (at the beginning of the simulation) and 1 (once the simulation 
		has been run for :attr:`topo_time`).
	"""
	def __init__(self, x, topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		"""
		Constructor.

		Parameters
		----------
		x : obj
			:class:`~tasmania.grids.axis.Axis` representing the underlying horizontal axis.
		topo_type : `str`, optional
			Topography type. Either: 
			
				* 'flat_terrain' (default); 
				* 'gaussian';
				* 'user_defined'.

		topo_time : obj
			class:`datetime.timedelta` representing the elapsed simulation time after which the topography 
			should stop increasing. Default is 0, corresponding to a time-invariant terrain surface-height.

		Keyword arguments
		-----------------
		topo_max_height : float
			When :data:`topo_type` is 'gaussian', maximum mountain height (in meters). Default is 500.
		topo_width_x : float
			When :data:`topo_type` is 'gaussian', mountain half-width (in meters). Default is 10000.
		topo_str : str
			When :data:`topo_type` is 'user_defined', terrain profile expression in the independent variable :math:`x`. 
			Must be fully C++-compliant.
		topo_smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise. Default is :obj:`False`.
		"""
		if topo_type not in ['flat_terrain', 'gaussian', 'user_defined']:
			raise ValueError("""Unknown topography type. Supported types are:
							 ''flat_terrain'', ''gaussian'', or ''user_defined''.""")

		self.topo_type = topo_type
		self.topo_time = topo_time
		self.topo_fact = float(self.topo_time == timedelta())

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((x.values.shape[0]), dtype = datatype)
		elif self.topo_type == 'gaussian':
			topo_max_height = kwargs.get('topo_max_height', 500.)
			topo_width_x = kwargs.get('topo_width_x', 10000.)
			xv = x.values
			c = 0.5 * (xv[0] + xv[-1])

			self._topo_final = topo_max_height * np.exp(- ((xv - c) / topo_width_x)**2)
		elif self.topo_type == 'user_defined':
			topo_str = kwargs.get('topo_str', 'x').encode('UTF-8')
			
			try:
				from grids.parser.parser_1d import Parser1d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			parser = Parser1d(topo_str, x.values)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		if kwargs.get('topo_smooth', False):
			self._topo_final[1:-1] += 0.25 * (self._topo_final[:-2] - 2. * self._topo_final[1:-1] + self._topo_final[2:]) 

		self.topo = xr.DataArray(self.topo_fact * self._topo_final, 
								 coords = x.coords, dims = x.dims, attrs = {'units': 'm'})	
		
	def update(self, time):
		"""
		Update topography at current simulation time.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		if lt(self.topo_fact, 1.):
			self.topo_fact = min(time / self.topo_time, 1.)
			self.topo.values = self.topo_fact * self._topo_final
		

class Topography2d:
	"""
	Class representing a two-dimensional topography.

	Attributes
	----------
	topo : array_like
		:class:`xarray.DataArray` representing the topography (in meters).
	topo_type : str
		Topography type. Either: 
		
			* 'flat_terrain';
			* 'gaussian'; 
			* 'schaer';
			* 'user_defined'.

	topo_time : obj
		:class:`datetime.timedelta` representing the elapsed simulation time after which the topography 
		should stop increasing.
	topo_fact : float
		Topography factor. It runs in between 0 (at the beginning of the simulation) and 1 (once the simulation 
		has been run for :attr:`topo_time`).
	topo_kwargs : dict
		Dictionary storing all the topography settings which could be passed to the constructor as keyword arguments.
	"""
	def __init__(self, grid, topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~tasmania.grids.grid_xy.GridXY` representing the underlying grid.
		topo_type : `str`, optional
			Topography type. Either:
			
				* 'flat_terrain' (default);
				* 'gaussian';
				* 'schaer'; 
				* 'user_defined'.

		topo_time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time after which the topography 
			should stop increasing. Default is 0, corresponding to a time-invariant terrain surface-height.

		Keyword arguments
		-----------------
		topo_max_height : float
			When :data:`topo_type` is either 'gaussian' or 'schaer', maximum mountain height (in meters). 
			Default is 500.
		topo_center_x : float
			When :data:`topo_type` is either 'gaussian' or 'schaer', :math:`x`-coordinate of the mountain center 
			(in meters). By default, the mountain center is placed in the center of the domain.
		topo_center_y : float
			When :data:`topo_type` is either 'gaussian' or 'schaer', :math:`y`-coordinate of the mountain center 
			(in meters). By default, the mountain center is placed in the center of the domain.
		topo_width_x : float
			When :data:`topo_type` is either 'gaussian' or 'schaer', mountain half-width in :math:`x`-direction 
			(in meters). Default is 10000.
		topo_width_y : float
			When :data:`topo_type` is either 'gaussian' or 'schaer', mountain half-width in	:math:`y`-direction 
			(in meters). Default is 10000.
		topo_str : str
			When :data:`topo_type` is 'user_defined', terrain profile expression in the independent variables 
			:math:`x` and :math:`y`. Must be fully C++-compliant.
		topo_smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise. Default is :obj:`False`.
		"""
		if topo_type not in ['flat_terrain', 'gaussian', 'schaer', 'user_defined']:
			raise ValueError("""Unknown topography type. Supported types are: \n"""
							 """''flat_terrain'', ''gaussian'', ''schaer'', or ''user_defined''.""")

		self.topo_type = topo_type
		self.topo_time = topo_time
		self.topo_fact = float(self.topo_time == timedelta())
		self.topo_kwargs = kwargs

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((grid.nx, grid.ny), dtype = datatype)
		elif self.topo_type == 'gaussian':
			# Shortcuts
			xv, yv = grid.x.values, grid.y.values
			nx, ny = grid.nx, grid.ny
			cx, cy = 0.5 * (xv[0] + xv[-1]), 0.5 * (yv[0] + yv[-1])

			# Set topography settings
			topo_max_height = 500. if kwargs.get('topo_max_height') is None else kwargs['topo_max_height']
			topo_center_x   = cx if kwargs.get('topo_center_x') is None else kwargs['topo_center_x']
			topo_center_y   = cy if kwargs.get('topo_center_y') is None else kwargs['topo_center_y']
			topo_width_x    = 10000. if kwargs.get('topo_width_x') is None else kwargs['topo_width_x']
			topo_width_y    = 10000. if kwargs.get('topo_width_y') is None else kwargs['topo_width_y']

			# Update settings list
			self.topo_kwargs['topo_max_height'] = topo_max_height
			self.topo_kwargs['topo_center_x']   = topo_center_x
			self.topo_kwargs['topo_center_y']   = topo_center_y
			self.topo_kwargs['topo_width_x']    = topo_width_x
			self.topo_kwargs['topo_width_y']    = topo_width_y

			# Compute topography profile
			x, y = np.meshgrid(xv, yv, indexing = 'ij')
			self._topo_final = topo_max_height * np.exp(- ((x - topo_center_x) / topo_width_x)**2 
														- ((y - topo_center_y) / topo_width_y)**2)
		elif self.topo_type == 'schaer':
			# Shortcuts
			xv, yv = grid.x.values, grid.y.values
			nx, ny = grid.nx, grid.ny
			cx, cy = 0.5 * (xv[0] + xv[-1]), 0.5 * (yv[0] + yv[-1])

			# Set topography settings
			topo_max_height = 500. if kwargs.get('topo_max_height') is None else kwargs['topo_max_height']
			topo_center_x   = cx if kwargs.get('topo_center_x') is None else kwargs['topo_center_x']
			topo_center_y   = cy if kwargs.get('topo_center_y') is None else kwargs['topo_center_y']
			topo_width_x    = 10000. if kwargs.get('topo_width_x') is None else kwargs['topo_width_x']
			topo_width_y    = 10000. if kwargs.get('topo_width_y') is None else kwargs['topo_width_y']

			# Update settings list
			self.topo_kwargs['topo_max_height'] = topo_max_height
			self.topo_kwargs['topo_center_x']   = topo_center_x
			self.topo_kwargs['topo_center_y']   = topo_center_y
			self.topo_kwargs['topo_width_x']    = topo_width_x
			self.topo_kwargs['topo_width_y']    = topo_width_y

			# Compute topography profile
			x, y = np.meshgrid(xv, yv, indexing = 'ij')
			self._topo_final = topo_max_height / ((1 + ((x - topo_center_x) / topo_width_x)**2 + 
													   ((y - topo_center_y) / topo_width_y)**2) ** 1.5)
		elif self.topo_type == 'user_defined':
			# Set topography expression
			topo_str = 'x + y'.encode('UTF-8') if kwargs.get('topo_str') is None else kwargs['topo_str'].encode('UTF-8')

			# Update settings list
			self.topo_kwargs['topo_str'] = topo_str
			
			# Import the parser
			try:
				from grids.parser.parser_2d import Parser2d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			# Parse
			parser = Parser2d(topo_str, grid.x.values, grid.y.values)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		if kwargs.get('topo_smooth', False):
			self._topo_final[1:-1, 1:-1] += 0.125 * (self._topo_final[:-2, 1:-1] + self._topo_final[2:, 1:-1] +
													 self._topo_final[1:-1, :-2] + self._topo_final[1:-1, 2:] -
													 4. * self._topo_final[1:-1, 1:-1]) 

		self.topo = xr.DataArray(self.topo_fact * self._topo_final, 
								 coords = [grid.x.values, grid.y.values], 
								 dims = [grid.x.dims[0], grid.y.dims[0]], attrs = {'units': 'm'})	
		
	def update(self, time):
		"""
		Update topography at current simulation time.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		if lt(self.topo_fact, 1.):
			self.topo_fact = min(time / self.topo_time, 1.)
			self.topo.values = self.topo_fact * self._topo_final

	def plot(self, grid, **kwargs):
		"""
		Plot the topography using the `mplot3d toolkit <https://matplotlib.org/tutorials/toolkits/mplot3d.html>`_.

		Parameters
		----------
		grid : obj
			:class:`~tasmania.grids.grid_xy.GridXY` representing the underlying grid.

		Keyword arguments
		-----------------
		**kwargs :
			Keyword arguments to be forwarded to :func:`matplotlib.pyplot.figure`.
		"""
		nx, ny = grid.nx, grid.ny
		xv, yv = grid.x.values, grid.y.values
		xv = np.repeat(xv[:,np.newaxis], ny, axis = 1)
		yv = np.repeat(np.reshape(yv[:,np.newaxis], (1, ny)), nx, axis = 0)

		if not kwargs:
			kwargs = {'figsize': [11,8]}
		fig = plt.figure(**kwargs)
		ax = fig.gca(projection = '3d')
		
		surf = ax.plot_surface(xv, yv, self.topo.values, 
			cmap = cm.coolwarm, linewidth = .1)

		ax.set(xlabel = '$x$', ylabel = '$y$', zlabel = '$h_s(x,y)$')
		fig.colorbar(surf)

		plt.show()
		
