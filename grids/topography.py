from datetime import timedelta
import numpy as np
import sympl

from tasmania.namelist import datatype
from tasmania.utils.utils import smaller_than as lt


class Topography1d:
	"""
    Class which represents a one-dimensional topography, possibly time-dependent.
    Indeed, although clearly not physical, a terrain surface (slowly) growing in
    the early stages of a simulation may help to retrieve numerical stability,
    as it prevents steep gradients in the first few iterations.

    Letting :math:`h_s = h_s(x)` be a one-dimensional topography, with
    :math:`x \in [a,b]`, the user may choose among:

        * a flat terrain, i.e., :math:`h_s(x) \equiv 0`;
        * a Gaussian-shaped mountain, i.e.,

            .. math::
                h_s(x) = h_{max} \exp{\left[ - \left( \\frac{x - c}{\sigma_x}
                \\right)^2 \\right]},

          where :math:`c = 0.5 (a + b)`.

    Further, user-defined profiles are supported as well, provided that they
    admit an analytical expression. This is passed to the class as a string,
    which is then parsed in C++ via `Cython <http://cython.org>`_
    (see :class:`~tasmania.grids.parser.parser_1d`). Therefore, the string
    must be fully C++-compliant.

	Attributes
	----------
	topo : dataarray_like
		:class:`sympl.DataArray` representing the topography (in meters).
	topo_type : str
		Topography type. Either:

			* 'flat_terrain'; 
			* 'gaussian';
			* 'user_defined'.

	topo_time : timedelta
		:class:`datetime.timedelta` object representing the elapsed
		simulation time after which the topography should stop increasing.
	topo_fact : float
		Topography factor. It runs in between 0 (at the beginning of the simulation)
		and 1 (once the simulation has been run for :attr:`topo_time`).
	topo_kwargs : dict
		Dictionary storing all the topography settings which could be passed
		to the constructor as keyword arguments.
	"""
	def __init__(self, x, topo_type='flat_terrain', topo_time=timedelta(),
				 dtype=datatype, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		x : dataarray_like
			:class:`sympl.DataArray` representing the underlying horizontal axis.
		topo_type : `str`, optional
			Topography type. Either: 
			
				* 'flat_terrain' (default); 
				* 'gaussian';
				* 'user_defined'.

		topo_time : timedelta
			class:`datetime.timedelta` representing the elapsed simulation time
			after which the topography should stop increasing. Default is 0,
			corresponding to a time-invariant terrain surface-height.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for any
			:class:`numpy.ndarray`. Defaults to :obj:`tasmania.namelist.datatype`.

		Keyword arguments
		-----------------
		topo_max_height : float
			When :data:`topo_type` is 'gaussian', maximum mountain height (in meters).
			Default is 500.
		topo_width_x : float
			When :data:`topo_type` is 'gaussian', mountain half-width (in meters).
			Default is 10000.
		topo_str : str
			When :data:`topo_type` is 'user_defined', terrain profile expression
			in the independent variable :math:`x`. Must be fully C++-compliant.
		topo_smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise.
			Default is :obj:`False`.

		Raises
		------
		ValueError :
			If the argument :obj:`topo_type` is neither 'flat_terrain',
			'gaussian', nor 'user_defined'.
		ImportError :
			If :class:`~tasmania.grid.parser.parser_1d.Parser1d` cannot be
			imported (likely because it has not been compiled).
		"""
		if topo_type not in ['flat_terrain', 'gaussian', 'user_defined']:
			raise ValueError("""Unknown topography type. Supported types are:
							 ''flat_terrain'', ''gaussian'', or ''user_defined''.""")

		self.topo_type   = topo_type
		self.topo_time   = topo_time
		self.topo_fact   = float(self.topo_time == timedelta())
		self.topo_kwargs = kwargs

		# Convert the input axis in meters, if necessary
		_x = x.to_units('m')

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((x.values.shape[0]), dtype=dtype)
		elif self.topo_type == 'gaussian':
			topo_max_height = kwargs.get('topo_max_height', 500.)
			topo_width_x = kwargs.get('topo_width_x', 10000.)
			xv = _x.values
			c = 0.5 * (xv[0] + xv[-1])

			self._topo_final = topo_max_height * np.exp(-((xv-c) / topo_width_x)**2)
		elif self.topo_type == 'user_defined':
			topo_str = kwargs.get('topo_str', 'x').encode('UTF-8')
			
			try:
				from tasmania.grids.parser.parser_1d import Parser1d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			parser = Parser1d(topo_str, _x.values)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		if kwargs.get('topo_smooth', False):
			self._topo_final[1:-1] += 0.25 * (self._topo_final[:-2] -
											  2.*self._topo_final[1:-1] +
											  self._topo_final[2:])

		self.topo = sympl.DataArray(self.topo_fact*self._topo_final,
									coords=x.coords, dims=x.dims, attrs={'units': 'm'})
		
	def update(self, time):
		"""
		Update topography at current simulation time.

		Parameters
		----------
		time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		if lt(self.topo_fact, 1.):
			self.topo_fact = min(time/self.topo_time, 1.)
			self.topo.values = self.topo_fact * self._topo_final


class Topography2d:
	"""
    Class which represents a two-dimensional topography, possibly time-dependent.
    Indeed, although clearly not physical, a terrain surface (slowly) growing in
    the early stages of a simulation may help to retrieve numerical stability,
    as it prevents steep gradients in the first few iterations.

    Letting :math:`h_s = h_s(x,y)` be a two-dimensional topography,
    with :math:`x \in [a_x,b_x]` and :math:`y \in [a_y,b_y]`, the user may
    choose among:

        * a flat terrain, i.e., :math:`h_s(x,y) \equiv 0`;
        * a Gaussian shaped-mountain, i.e.

            .. math::
                h_s(x,y) = h_{max} \exp{\left[ - \left( \\frac{x - c_x}{\sigma_x}
                \\right)^2 - \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]} ;

        * a modified Gaussian-shaped mountain proposed by Schaer and Durran (1997),

            .. math::
                h_s(x,y) = \\frac{h_{max}}{\left[ 1 + \left( \\frac{x - c_x}{\sigma_x}
                \\right)^2 + \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]^{3/2}}.

    Further, user-defined profiles are supported as well, provided that they
    admit an analytical expression. This is passed to the class as a string,
    which is then parsed in C++ via `Cython <http://cython.org>`_
    (see :class:`~tasmania.grids.parser.parser_2d`). Therefore, the string
    must be fully C++-compliant.

    Reference
    ---------
	Schaer, C., and D. R. Durran. (1997). Vortex formation and vortex shedding \
	in continuosly stratified flows past isolated topography. \
	*Journal of Atmospheric Sciences*, *54*:534-554.

	Attributes
	----------
	topo : dataarray_like
		:class:`sympl.DataArray` representing the topography (in meters).
	topo_type : str
		Topography type. Either: 
		
			* 'flat_terrain';
			* 'gaussian'; 
			* 'schaer';
			* 'user_defined'.

	topo_time : timedelta
		:class:`datetime.timedelta` representing the elapsed simulation time
		after which the topography should stop increasing.
	topo_fact : float
		Topography factor. It runs in between 0 (at the beginning of the simulation)
		and 1 (once the simulation has been run for :attr:`topo_time`).
	topo_kwargs : dict
		Dictionary storing all the topography settings which could be passed
		to the constructor as keyword arguments.
	"""
	def __init__(self, grid, topo_type='flat_terrain', topo_time=timedelta(),
				 dtype=datatype, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xy.GridXY` representing the underlying grid.
		topo_type : `str`, optional
			Topography type. Either:
			
				* 'flat_terrain' (default);
				* 'gaussian';
				* 'schaer'; 
				* 'user_defined'.

		topo_time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time
			after which the topography should stop increasing. Default is 0,
			corresponding to a time-invariant terrain surface-height.

		Keyword arguments
		-----------------
		topo_max_height : float
			When :data:`topo_type` is either 'gaussian' or 'schaer',
			maximum mountain height (in meters). Default is 500.
		topo_center_x : float
			When :data:`topo_type` is either 'gaussian' or 'schaer',
			:math:`x`-coordinate of the mountain center (in meters).
			By default, the mountain center is placed in the center of the domain.
		topo_center_y : float
			When :data:`topo_type` is either 'gaussian' or 'schaer',
			:math:`y`-coordinate of the mountain center (in meters).
			By default, the mountain center is placed in the center of the domain.
		topo_width_x : float
			When :data:`topo_type` is either 'gaussian' or 'schaer',
			mountain half-width in :math:`x`-direction (in meters). Default is 10000.
		topo_width_y : float
			When :data:`topo_type` is either 'gaussian' or 'schaer',
			mountain half-width in	:math:`y`-direction (in meters). Default is 10000.
		topo_str : str
			When :data:`topo_type` is 'user_defined', terrain profile expression
			in the independent variables :math:`x` and :math:`y`.
			Must be fully C++-compliant.
		topo_smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise.
			Default is :obj:`False`.

		Raises
		------
		ValueError :
			If the argument :obj:`topo_type` is neither 'flat_terrain',
			'gaussian', 'schaer', nor 'user_defined'.
		ImportError :
			If :class:`~tasmania.grid.parser.parser_2d.Parser2d` cannot be
			imported (likely because it has not been compiled).
		"""
		if topo_type not in ['flat_terrain', 'gaussian', 'schaer', 'user_defined']:
			raise ValueError("""Unknown topography type. Supported types are: \n
							    ''flat_terrain'', ''gaussian'', ''schaer'', or 
							    ''user_defined''.""")

		self.topo_type   = topo_type
		self.topo_time   = topo_time
		self.topo_fact   = float(self.topo_time == timedelta())
		self.topo_kwargs = kwargs

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((grid.nx, grid.ny), dtype=dtype)
		elif self.topo_type == 'gaussian':
			# Convert axes into meters, if necessary
			_x, _y = grid.x.to_units('m'), grid.y.to_units('m')

			# Shortcuts
			xv, yv = _x.values, _y.values
			cx, cy = 0.5 * (xv[0]+xv[-1]), 0.5 * (yv[0]+yv[-1])

			# Set topography settings
			topo_max_height = 500. if kwargs.get('topo_max_height') is None \
							  else kwargs['topo_max_height']
			topo_center_x   = cx if kwargs.get('topo_center_x') is None \
							  else kwargs['topo_center_x']
			topo_center_y   = cy if kwargs.get('topo_center_y') is None \
							  else kwargs['topo_center_y']
			topo_width_x    = 10000. if kwargs.get('topo_width_x') is None \
							  else kwargs['topo_width_x']
			topo_width_y    = 10000. if kwargs.get('topo_width_y') is None \
							  else kwargs['topo_width_y']

			# Update settings list
			self.topo_kwargs['topo_max_height'] = topo_max_height
			self.topo_kwargs['topo_center_x']   = topo_center_x
			self.topo_kwargs['topo_center_y']   = topo_center_y
			self.topo_kwargs['topo_width_x']    = topo_width_x
			self.topo_kwargs['topo_width_y']    = topo_width_y

			# Compute topography profile
			x, y = np.meshgrid(xv, yv, indexing='ij')
			self._topo_final = topo_max_height * \
							   np.exp(- ((x-topo_center_x) / topo_width_x)**2
									  - ((y-topo_center_y) / topo_width_y)**2)
		elif self.topo_type == 'schaer':
			# Convert axes into meters, if necessary
			_x, _y = grid.x.to_units('m'), grid.y.to_units('m')

			# Shortcuts
			xv, yv = _x.values, _y.values
			cx, cy = 0.5 * (xv[0]+xv[-1]), 0.5 * (yv[0]+yv[-1])

			# Set topography settings
			topo_max_height = 500. if kwargs.get('topo_max_height') is None \
							  else kwargs['topo_max_height']
			topo_center_x   = cx if kwargs.get('topo_center_x') is None \
							  else kwargs['topo_center_x']
			topo_center_y   = cy if kwargs.get('topo_center_y') is None \
							  else kwargs['topo_center_y']
			topo_width_x    = 10000. if kwargs.get('topo_width_x') is None \
							  else kwargs['topo_width_x']
			topo_width_y    = 10000. if kwargs.get('topo_width_y') is None \
							  else kwargs['topo_width_y']

			# Update settings list
			self.topo_kwargs['topo_max_height'] = topo_max_height
			self.topo_kwargs['topo_center_x']   = topo_center_x
			self.topo_kwargs['topo_center_y']   = topo_center_y
			self.topo_kwargs['topo_width_x']    = topo_width_x
			self.topo_kwargs['topo_width_y']    = topo_width_y

			# Compute topography profile
			x, y = np.meshgrid(xv, yv, indexing='ij')
			self._topo_final = topo_max_height / \
							   ((1 + ((x-topo_center_x) / topo_width_x)**2 +
									 ((y-topo_center_y) / topo_width_y)**2) ** 1.5)
		elif self.topo_type == 'user_defined':
			# Set topography expression
			topo_str = 'x + y'.encode('UTF-8') if kwargs.get('topo_str') is None \
					   else kwargs['topo_str'].encode('UTF-8')

			# Update settings list
			self.topo_kwargs['topo_str'] = topo_str
			
			# Import the parser
			try:
				from tasmania.grids.parser.parser_2d import Parser2d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			# Parse
			parser = Parser2d(topo_str, grid.x.to_units('m').values,
							  grid.y.to_units('m').values)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		if kwargs.get('topo_smooth', False):
			self._topo_final[1:-1, 1:-1] += 0.125 * (self._topo_final[:-2, 1:-1] +
													 self._topo_final[2:, 1:-1] +
													 self._topo_final[1:-1, :-2] +
													 self._topo_final[1:-1, 2:] -
													 4.*self._topo_final[1:-1, 1:-1]) 

		self.topo = sympl.DataArray(self.topo_fact*self._topo_final,
									coords=[grid.x.values, grid.y.values],
								 	dims=[grid.x.dims[0], grid.y.dims[0]],
									attrs={'units': 'm'})
		
	def update(self, time):
		"""
		Update topography at current simulation time.

		Parameters
		----------
		time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		if lt(self.topo_fact, 1.):
			self.topo_fact = min(time/self.topo_time, 1.)
			self.topo.values = self.topo_fact * self._topo_final
