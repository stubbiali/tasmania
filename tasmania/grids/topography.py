"""
This module contains:
	Topography{1d, 2d}
"""
from copy import deepcopy
from datetime import timedelta
import numpy as np
from sympl import DataArray

from tasmania.utils.utils import smaller_than as lt
try:
	from namelist import datatype
except ImportError:
	datatype = np.float32


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
		1-D :class:`sympl.DataArray` representing the topography ([m]).
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
			1-D :class:`sympl.DataArray` representing the underlying horizontal axis.
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
		topo_max_height : dataarray_like
			1-item :class:`sympl.DataArray` representing the maximum mountain height.
			Defaults to 500 m. Effective only when :data:`topo_type` is 'gaussian'.
		topo_center_x : dataarray_like
			1-item :class:`sympl.DataArray` representing the :math:`x`-coordinate of
			the mountain center. By default, the mountain center is placed in the center
			of the domain. Effective only when :data:`topo_type` is 'gaussian'.
		topo_width_x : dataarray_like
			1-item :class:`sympl.DataArray` representing the mountain half-width.
			Defaults to 1, in the same units of :data:`x`. Effective only when
			:data:`topo_type` is 'gaussian'.
		topo_str : str
			Terrain profile expression in the independent variable :math:`x`.
			Must be fully C++-compliant. Effective only when :data:`topo_type`
			is 'user_defined'.
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
		self.topo_kwargs = deepcopy(kwargs)

		xv = x.values

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((x.values.shape[0]), dtype=dtype)
		elif self.topo_type == 'gaussian':
			topo_max_height_ = kwargs.get('topo_max_height',
										  DataArray(500.0, attrs={'units': 'm'}))
			topo_max_height = topo_max_height_.to_units('m').values.item()

			topo_width_x_ = kwargs.get('topo_width_x',
									  DataArray(1.0, attrs={'units': x.attrs['units']}))
			topo_width_x = topo_width_x_.to_units(x.attrs['units']).values.item()

			cx = 0.5 * (xv[0] + xv[-1])
			topo_center_x = cx if kwargs.get('topo_center_x') is None else \
				kwargs['topo_center_x'].to_units(x.attrs['units']).values.item()

			self.topo_kwargs['topo_max_height'] = \
				DataArray(topo_max_height, attrs={'units': 'm'})
			self.topo_kwargs['topo_width_x'] = \
				DataArray(topo_width_x, attrs={'units': x.attrs['units']})
			self.topo_kwargs['topo_center_x'] = \
				DataArray(topo_center_x, attrs={'units': x.attrs['units']})

			self._topo_final = topo_max_height * np.exp(- ((xv-cx) / topo_width_x)**2)
		elif self.topo_type == 'user_defined':
			topo_str = kwargs.get('topo_str', 'x')

			self.topo_kwargs['topo_str'] = topo_str
			
			try:
				from tasmania.grids.parser.parser_1d import Parser1d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			parser = Parser1d(topo_str.encode('UTF-8'), xv)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		self.topo_kwargs['topo_smooth'] = kwargs.get('topo_smooth', False)
		if self.topo_kwargs['topo_smooth']:
			self._topo_final[1:-1] += 0.25 * (self._topo_final[:-2] -
											  2.*self._topo_final[1:-1] +
											  self._topo_final[2:])

		self.topo = DataArray(self.topo_fact*self._topo_final,
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
		2-D :class:`sympl.DataArray` representing the topography ([m]).
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
		topo_max_height : dataarray_like
			1-item :class:`sympl.DataArray` representing the maximum mountain height.
			Defaults to 500. Effective when :data:`topo_type` is either 'gaussian' or 'schaer'.
		topo_center_x : dataarray_like
			1-item :class:`sympl.DataArray` representing the :math:`x`-coordinate of
			the mountain center. By default, the mountain center is placed in the center
			of the domain. Effective when :data:`topo_type` is either 'gaussian' or 'schaer'.
		topo_center_y : dataarray_like
			1-item :class:`sympl.DataArray` representing the :math:`y`-coordinate of
			the mountain center. By default, the mountain center is placed in the center
			of the domain. Effective when :data:`topo_type` is either 'gaussian' or 'schaer'.
		topo_width_x : dataarray_like
			1-item :class:`sympl.DataArray` representing the mountain half-width in
			the :math:`x`-direction. Defaults to 1, in the same units of the :data:`x`-axis.
			Effective when :data:`topo_type` is either 'gaussian' or 'schaer'.
		topo_width_y : dataarray_like
			1-item :class:`sympl.DataArray` representing the mountain half-width in
			the :math:`y`-direction. Defaults to 1, in the same units of the :data:`y`-axis.
			Effective when :data:`topo_type` is either 'gaussian' or 'schaer'.
		topo_str : str
			Terrain profile expression in the independent variables :math:`x` and :math:`y`.
			Must be fully C++-compliant. Effective only when :data:`topo_type`
			is 'user_defined'.
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
		self.topo_kwargs = deepcopy(kwargs)

		x, y = grid.x, grid.y
		xv, yv = grid.x.values, grid.y.values

		if self.topo_type == 'flat_terrain':
			self._topo_final = np.zeros((grid.nx, grid.ny), dtype=dtype)
		elif self.topo_type == 'gaussian':
			topo_max_height_ = kwargs.get('topo_max_height',
										  DataArray(500.0, attrs={'units': 'm'}))
			topo_max_height = topo_max_height_.to_units('m').values.item()

			topo_width_x_ = kwargs.get('topo_width_x',
									   DataArray(1.0, attrs={'units': x.attrs['units']}))
			topo_width_x = topo_width_x_.to_units(x.attrs['units']).values.item()

			topo_width_y_ = kwargs.get('topo_width_y',
									   DataArray(1.0, attrs={'units': y.attrs['units']}))
			topo_width_y = topo_width_y_.to_units(y.attrs['units']).values.item()

			cx = 0.5 * (xv[0] + xv[-1])
			topo_center_x = cx if kwargs.get('topo_center_x') is None else \
							kwargs['topo_center_x'].to_units(x.attrs['units']).values.item()

			cy = 0.5 * (yv[0] + yv[-1])
			topo_center_y = cy if kwargs.get('topo_center_y') is None else \
				kwargs['topo_center_y'].to_units(y.attrs['units']).values.item()

			self.topo_kwargs['topo_max_height'] = \
				DataArray(topo_max_height, attrs={'units': 'm'})
			self.topo_kwargs['topo_width_x'] = \
				DataArray(topo_width_x, attrs={'units': x.attrs['units']})
			self.topo_kwargs['topo_width_y'] = \
				DataArray(topo_width_y, attrs={'units': y.attrs['units']})
			self.topo_kwargs['topo_center_x'] = \
				DataArray(topo_center_x, attrs={'units': x.attrs['units']})
			self.topo_kwargs['topo_center_y'] = \
				DataArray(topo_center_y, attrs={'units': y.attrs['units']})

			xv_, yv_ = np.meshgrid(xv, yv, indexing='ij')
			self._topo_final = topo_max_height * \
							   np.exp(- ((xv_ - topo_center_x) / topo_width_x)**2
									  - ((yv_ - topo_center_y) / topo_width_y)**2)
		elif self.topo_type == 'schaer':
			topo_max_height_ = kwargs.get('topo_max_height',
										  DataArray(500.0, attrs={'units': 'm'}))
			topo_max_height = topo_max_height_.to_units('m').values.item()

			topo_width_x_ = kwargs.get('topo_width_x',
									   DataArray(1.0, attrs={'units': x.attrs['units']}))
			topo_width_x = topo_width_x_.to_units(x.attrs['units']).values.item()

			topo_width_y_ = kwargs.get('topo_width_y',
									   DataArray(1.0, attrs={'units': y.attrs['units']}))
			topo_width_y = topo_width_y_.to_units(y.attrs['units']).values.item()

			cx = 0.5 * (xv[0] + xv[-1])
			topo_center_x = cx if kwargs.get('topo_center_x') is None else \
				kwargs['topo_center_x'].to_units(x.attrs['units']).values.item()

			cy = 0.5 * (yv[0] + yv[-1])
			topo_center_y = cy if kwargs.get('topo_center_y') is None else \
				kwargs['topo_center_y'].to_units(y.attrs['units']).values.item()

			self.topo_kwargs['topo_max_height'] = \
				DataArray(topo_max_height, attrs={'units': 'm'})
			self.topo_kwargs['topo_width_x'] = \
				DataArray(topo_width_x, attrs={'units': x.attrs['units']})
			self.topo_kwargs['topo_width_y'] = \
				DataArray(topo_width_y, attrs={'units': y.attrs['units']})
			self.topo_kwargs['topo_center_x'] = \
				DataArray(topo_center_x, attrs={'units': x.attrs['units']})
			self.topo_kwargs['topo_center_y'] = \
				DataArray(topo_center_y, attrs={'units': y.attrs['units']})

			xv_, yv_ = np.meshgrid(xv, yv, indexing='ij')
			self._topo_final = topo_max_height / \
							   ((1 + ((xv_ - topo_center_x) / topo_width_x)**2 +
									 ((yv_ - topo_center_y) / topo_width_y)**2) ** 1.5)
		elif self.topo_type == 'user_defined':
			topo_str = 'x + y' if kwargs.get('topo_str') is None else kwargs['topo_str']

			self.topo_kwargs['topo_str'] = topo_str
			
			# Import the parser
			try:
				from tasmania.grids.parser.parser_2d import Parser2d
			except ImportError:
				print('Hint: did you compile the parser?')
				raise
				
			# Parse
			parser = Parser2d(topo_str.encode('UTF-8'), grid.x.to_units('m').values,
							  grid.y.to_units('m').values)
			self._topo_final = parser.evaluate()
			
		# Smooth the topography out
		self.topo_kwargs['topo_smooth'] = kwargs.get('topo_smooth', False)
		if self.topo_kwargs['topo_smooth']:
			self._topo_final[1:-1, 1:-1] += 0.125 * (self._topo_final[:-2, 1:-1] +
													 self._topo_final[2:, 1:-1] +
													 self._topo_final[1:-1, :-2] +
													 self._topo_final[1:-1, 2:] -
													 4.*self._topo_final[1:-1, 1:-1]) 

		self.topo = DataArray(self.topo_fact*self._topo_final,
							  coords=[xv, yv], dims=[x.dims[0], y.dims[0]],
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
