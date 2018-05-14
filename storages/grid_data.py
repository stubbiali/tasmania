import copy
import numpy as np
import xarray as xr

from tasmania.grids.axis import Axis
import tasmania.utils.utils as utils
import tasmania.utils.utils_plot as utils_plot

class GridData:
	"""
	Class storing and handling time-dependent variables defined on a grid. Ideally, this class should be used to 
	represent the state, or a sequence of states at different time levels, of a *generic* climate or meteorological model. 
	The model variables, in the shape of :class:`numpy.ndarray`\s, are passed to the constructor as keyword arguments. 
	After conversion to :class:`xarray.DataArray`\s, the variables are packed in a dictionary whose keys are the input keywords. 
	The class attribute :data:`units` lists, for any admissible keyword, the units in which the associated field should 
	be expressed. Any variable can be accessed via the accessor operator by specifying the corresponding 
	keyword. Other methods are provided to update the state, or to create a sequence of states (useful for animation purposes). 
	This class is designed to be as general as possible. Hence, it is not endowed with any method whose
	implementation depends on the variables actually stored by the class. This kind of methods might be provided by some 
	derived classes, each one representing the state of a *specific* model.

	Attributes
	----------
	grid : obj
		The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
	"""
	# Specify the units in which variables should be expressed
	units = {
		'accumulated_precipitation'									: 'mm'			 	,
		'air_density'												: 'kg m-3'		 	,
		'air_isentropic_density'									: 'kg m-2 K-1'    	,
		'air_pressure'												: 'Pa'            	,
		'air_pressure_on_interface_levels'							: 'Pa'            	,
		'air_temperature'											: 'K'             	,
		'cloud_liquid_water_isentropic_density'						: 'kg m-2 K-1'    	,
		'exner_function'											: 'm2 s-2 K-2'    	,
		'exner_function_on_interface_levels'						: 'm2 s-2 K-2'    	,
		'height'													: 'm'             	,
		'height_on_interface_levels'								: 'm'             	,
		'mass_fraction_of_cloud_liquid_water_in_air'				: 'kg kg-1'       	,
		'mass_fraction_of_precipitation_water_in_air'				: 'kg kg-1'       	,
		'mass_fraction_of_water_vapor_in_air'						: 'kg kg-1'       	,
		'montgomery_potential'										: 'm2 s-2'        	,
		'precipitation'												: 'mm h-1'        	,
		'precipitation_water_isentropic_density'					: 'kg m-2 K-1'    	,
		'raindrop_fall_speed'										: 'm s-1'		 	,
		'tendency_of_air_potential_temperature'						: 'K s-1'		 	,
		'tendency_of_mass_fraction_of_cloud_liquid_water_in_air'	: 'kg kg-1 s-1'		,
		'tendency_of_mass_fraction_of_precipitation_water_in_air'	: 'kg kg-1 s-1'		,
		'tendency_of_mass_fraction_of_water_vapor_in_air'			: 'kg kg-1 s-1'		,
		'water_vapor_isentropic_density'							: 'kg m-2 K-1'    	,
		'x_momentum_isentropic'										: 'kg m-1 s-1 K-1'	,
		'x_velocity'												: 'm s-1'         	,
		'x_velocity_unstaggered'									: 'm s-1'         	,
		'y_momentum_isentropic'										: 'kg m-1 s-1 K-1'	,
		'y_velocity'												: 'm s-1'         	,
		'y_velocity_unstaggered'									: 'm s-1'         	,
	}

	def __init__(self, time, grid, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		time : obj
			:class:`datetime.datetime` representing the time instant at which the variables are defined.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		**kwargs : array_like
			:class:`numpy.ndarray` representing a gridded variable.
		"""
		self.grid = grid

		self._vars = dict()
		for key in kwargs:
			var = kwargs[key]
			if var is not None:
				# Distinguish between horizontally staggered and unstaggered fields
				x = grid.x if var.shape[0] == grid.nx else grid.x_at_u_locations
				y = grid.y if var.shape[1] == grid.ny else grid.y_at_v_locations

				# Properly treat the vertical axis, so that either two- and three-dimensional arrays can be stored
				# A notable example of a two-dimensional field is the accumulated precipitation
				if len(var.shape) == 2:
					var = var[:, :, np.newaxis]
				if var.shape[2] == 1:
					z = Axis(np.array([grid.z_on_interface_levels[-1]]), grid.z.dims, attrs = grid.z.attrs)
				elif var.shape[2] == 1:
					z = Axis(np.array([grid.z_on_interface_levels[-1]]), grid.z.dims, attrs = grid.z.attrs)
				elif var.shape[2] == grid.nz:
					z = grid.z 
				elif var.shape[2] == grid.nz + 1:
					z = grid.z_on_interface_levels

				_var = xr.DataArray(var[:, :, :, np.newaxis], 
									coords = [x.values, y.values, z.values, [time]],
									dims = [x.dims, y.dims, z.dims, 'time'],
									attrs = {'units': GridData.units[key]})
				self._vars[key] = _var

	def __getitem__(self, key):
		"""
		Get a shallow copy of a gridded variable.

		Parameters
		----------
		key : str
			The name of the variable to return.

		Return
		------
		obj :
			Shallow copy of the :class:`xarray.DataArray` representing the variable, or :obj:`None` if the variable is not found.
		"""
		return self._vars.get(key, None)

	@property
	def variable_names(self):
		"""
		Get the names of the stored variables.

		Return
		------
		list :
			List of the names of the stored variables.
		"""
		return list(self._vars.keys())

	@property
	def time(self):
		"""
		Shortcut to the time instant at which the variables are defined.

		Return
		------
		obj :
			:class:`datetime.timedelta` representing the time instant at which the variables are defined.

		Warning
		-------
		Within an instance of this class, variables are not forced to be defined at the same time level, 
		so the behaviour of this method might be undefined.
		"""
		return utils.convert_datetime64_to_datetime(self._vars[self.variable_names[0]].coords['time'].values[0])

	def add_variables(self, time, **kwargs):
		"""
		Add a list of variables, passed as keyword arguments.

		Parameters
		----------
		time : obj
			:class:`datetime.datetime` representing the time instant at which the variables are defined.
		**kwargs : array_like
			:class:`numpy.ndarray` representing a gridded variable.
		"""
		for key in kwargs:
			var = kwargs[key]
			if var is not None:
				# Distinguish between horizontally staggered and unstaggered fields
				x = self.grid.x if var.shape[0] == self.grid.nx else self.grid.x_at_u_locations
				y = self.grid.y if var.shape[1] == self.grid.ny else self.grid.y_at_v_locations

				# Properly treat the vertical axis, so that either two- and three-dimensional arrays can be stored
				# A notable example of a two-dimensional field is the accumulated precipitation
				if len(var.shape) == 2:
					var = var[:, :, np.newaxis]
				if var.shape[2] == 1:
					z = Axis(np.array([self.grid.z_on_interface_levels[-1]]), self.grid.z.dims, attrs = self.grid.z.attrs)
				elif var.shape[2] == self.grid.nz:
					z = self.grid.z 
				elif var.shape[2] == self.grid.nz + 1:
					z = self.grid.z_on_interface_levels

				_var = xr.DataArray(var[:, :, :, np.newaxis], 
									coords = [x.values, y.values, z.values, [time]],
									dims = [x.dims, y.dims, z.dims, 'time'],
									attrs = {'units': GridData.units[key]})
				self._vars[key] = _var

	def extend(self, other):
		"""
		Extend the current object by adding the variables stored by another object.

		Notes
		-----
		* The variables are deep copied.
		* The incoming variables do not need to be defined at the same time level.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class) with which the current object will be synced.
		"""
		for key in other._vars:
			self._vars[key] = copy.deepcopy(other._vars[key])

	def update(self, other):
		"""
		Sync the current object with another :class:`~storages.grid_data.GridData` (or a derived class).
		
		Notes
		-----
		* It is assumed that *all* the variables stored by the input object are present also in the current object. 
		* After the update, the stored variables might not be all defined at the same time level.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class) with which the current object will be synced.
		"""
		for key in other._vars:
			self._vars[key].values[:,:,:,:] = other._vars[key].values[:,:,:,:]
			self._vars[key].coords['time']  = other._vars[key].coords['time']

	def extend_and_update(self, other):
		"""
		Sync the current object with another :class:`~storages.grid_data.GridData` (or a derived class).
		This implies that, for each variable stored in the incoming object:

		* if the current object contains a variable with the same name, that variable is updated;
		* if the current object does not contain any variable with the same name, that variable is deep copied inside \
			the current object.

		Note
		----
		After the update, the stored variables might not be all defined at the same time level.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class) with which the current object will be synced.
		"""
		for key in other._vars:
			try:
				self._vars[key].values[:,:,:,:] = other._vars[key].values[:,:,:,:]
				self._vars[key].coords['time'] = other._vars[key].coords['time']
			except KeyError:
				self._vars[key] = copy.deepcopy(other._vars[key])
	
	def pop(self, key):
		"""
		Get a shallow copy of a gridded variable, then remove it from the dictionary.

		Parameters
		----------
		key : str
			The name of the variable to return and remove.

		Return
		------
		obj :
			Shallow copy of the :class:`xarray.DataArray` representing the variable, or :obj:`None` if the variable is not found.
		"""
		try:
			return self._vars.pop(key, None)
		except KeyError:
			return None

	def append(self, other):
		"""
		Append a new state to the sequence of states.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class), whose :class:`xarray.DataArray`\s 
			will be concatenated along the temporal axis to the corresponding ones in the current object.

		Note
		----
		:data:`other` is supposed to contain exactly the same variables stored by the current object.
		"""
		for key in self._vars:
			self._vars[key] = xr.concat([self._vars[key], copy.deepcopy(other[key])], 'time')

	def get_max(self, key):
		"""
		Get the maximum value of a variable.

		Parameters
		----------
		key : str
			The name of the variable.

		Return
		------
		float :
			The maximum value of the variable of interest.
		"""
		if key in GridData.units.keys():
			return np.amax(self._vars[key].values[:,:,:,-1])
		else:
			raise KeyError('The variable {} is not stored within the current object'.format(key))

	def get_min(self, key):
		"""
		Get the minimum value of a variable.

		Parameters
		----------
		key : str
			The name of the variable.

		Return
		------
		float :
			The minimum value of the variable of interest.
		"""
		if key in GridData.units.keys():
			return np.amin(self._vars[key].values[:,:,:,-1])
		else:
			raise KeyError('The variable {} is not stored within the current object'.format(key))

	def __iadd__(self, other):
		"""
		"""
		for key in other._vars:
			if self._vars.get(key, None) is None:
				self._vars[key] = copy.deepcopy(other._vars[key])
			elif self._vars[key].coords['time'].values[0] != other._vars[key].coords['time'].values[0]:
				self._vars[key].coords['time'].values[0] = other._vars[key].coords['time'].values[0]
				self._vars[key].values[:,:,:,0] = other._vars[key].values[:,:,:,0]
			else:
				self._vars[key].values[:,:,:,0] += other._vars[key].values[:,:,:,0]

		return self

	def animation_profile_x(self, field_to_plot, y_level, z_level, destination, **kwargs):
		"""
		Generate an animation showing a field along a section line orthogonal to the :math:`yz`-plane.

		Parameters
		----------
		field_to_plot : str
			The name of the field to plot.	
		y_level : int
			:math:`y`-index identifying the section line. 
		z_level : int
			:math:`z`-index identifying the section line. 
		destination : str
			String specifying the path to the location where the movie will be saved. 
			Note that the string should include the extension as well.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`tasmania.utils.utils_plot.animation_profile_x` for the complete list.
		"""
		# Shortcuts
		nx = self.grid.nx
		time = self._vars[list(self._vars.keys())[0]].coords['time'].values 

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, z_level, :] 
		else:
			raise RuntimeError('Unknown field to plot.')

		# Infer the underlying x-grid
		x = self.grid.x.values if var.shape[0] == nx else self.grid.x_at_u_locations.values

		# Plot
		utils_plot.animation_profile_x(time, x, var, destination, **kwargs)
