from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
import xarray as xr

from namelist import datatype
from storages.grid_data import GridData
from utils import smaller_than as lt
from utils import reverse_colormap

class IsentropicState(GridData):
	"""
	This class inherits :class:`~storages.grid_data.GridData` to represent the state of the three-dimensional 
	(moist) isentropic model.
	"""
	def __init__(self, time, grid, isentropic_density, x_velocity, y_velocity, 
				 pressure, exner_function, montgomery_potential, height,
				 water_vapour = None, cloud_water = None, precipitation_water = None):
		"""
		Constructor.
		"""
		kwargs = {'isentropic_density'  : isentropic_density,
				  'x_velocity'          : x_velocity,
				  'y_velocity'          : y_velocity,
				  'pressure'            : pressure,
				  'exner_function'      : exner_function,
				  'montgomery_potential': montgomery_potential,
				  'height'              : height}
		if water_vapour is not None:
			kwargs['water_vapour'] = water_vapour
		if cloud_water is not None:
			kwargs['cloud_water'] = cloud_water
		if precipitation_water is not None:
			kwargs['precipitation_water'] = precipitation_water

		super().__init__(time, grid, **kwargs)

	def get_cfl(self, dt):
		"""
		Compute the CFL number. If this is greater than one, i.e., if the CFL condition is violated,
		the method throws a warning.

		Args:
			dt (obj): :class:`datetime.timedelta` object representing the time step.
		"""
		u_max = np.amax(np.absolute(self._vars['x_velocity'].values[:,:,:,-1]))
		v_max = np.amax(np.absolute(self._vars['y_velocity'].values[:,:,:,-1]))

		cfl_x = u_max * dt.seconds / self._grid.dx
		cfl_y = v_max * dt.seconds / self._grid.dy
		cfl = max(cfl_x, cfl_y)

		if cfl > 1.:
			raise RuntimeWarning('CFL condition violated.')
		elif np.isnan(cfl):
			raise RuntimeWarning('NaN values.')

		return cfl
	
	def plot_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Plot a field in the plane :math:`y = \bar{y}`.

		Args:
			field_to_plot (str): String specifying the model variable to plot.
			y_level (int): Index corresponding to the :math:`y`-level identifying 
				the :math:`(x,z)`-cross-section to plot.
			time_level (int): The index corresponding to the time level to plot.
			**kwargs: Keyword arguments defining various plotting settings.
		"""
		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, time_level]
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		nx, nz = self._grid.nx, self._grid.nz
		ni, nk = var.shape[0], var.shape[1]

		# Get keyword arguments
		title           = kwargs.get('title' , field_to_plot)
		x_label         = kwargs.get('x_label', '{} [${}$]'.format(self._grid.x.dims, self._grid.x.attrs.get('units', '')))
		x_factor        = kwargs.get('x_factor', 1.)
		z_label         = kwargs.get('z_label', '{} [${}$]'.format(self._grid.z.dims, self._grid.z.attrs.get('units', '')))
		z_factor        = kwargs.get('z_factor', 1.)
		cmap_name       = kwargs.get('cmap_name', 'RdYlBu')
		cmap_levels     = kwargs.get('cmap_levels', 31)
		cmap_center     = kwargs.get('cmap_center', None)
		cmap_half_width = kwargs.get('cmap_half_width', None)

		# The x-grid underlying the isentropes and the field
		x1 = x_factor * np.repeat(self._grid.x.values[:, np.newaxis], nz, axis = 1)
		xv = x_factor * (self._grid.x.values if ni == nx else self._grid.x_half_levels.values)
		x2 = np.repeat(xv[:, np.newaxis], nk, axis = 1)

		# The isentropes
		z = z_factor * self._vars['height'].values[:, y_level, :, time_level]
		z1 = z if nk == nz + 1 else 0.5 * (z[:, :-1] + z[:, 1:])

		# The z-grid underlying the field
		z2 = np.zeros((ni, nk), dtype = datatype)
		if ni == nx:
			z2[:, :] = z1[:, :]
		else:
			z2[1:-1, :] = 0.5 * (z1[:-1, :] + z1[1:, :])
			z2[0, :], z2[-1, :] = z2[1, :], z2[-2, :]

		# Instantiate figure and axis objects
		fig, ax = plt.subplots(figsize = [11,8])

		# Plot the isentropes
		for k in range(0, nk):
			ax.plot(x1[:, 0], z1[:, k], color = 'gray', linewidth = 1)
		ax.plot(x1[:, 0], z1[:, -1], color = 'black', linewidth = 1)

		# Create colormap
		field_min, field_max = np.amin(var), np.amax(var)
		if cmap_center is None or not (lt(field_min, cmap_center) and lt(cmap_center, field_max)):
			cmap_lb, cmap_ub = field_min, field_max
		else:
			half_width = max(cmap_center - field_min, field_max - cmap_center) if cmap_half_width is None else cmap_half_width
			cmap_lb, cmap_ub = cmap_center - half_width, cmap_center + half_width 
		color_scale = np.linspace(cmap_lb, cmap_ub, cmap_levels, endpoint = True)

		if cmap_name == 'BuRd':
			cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)

		# Plot the field
		surf = plt.contourf(x2, z2, var, color_scale, cmap = cm)

		# Set plot settings
		ax.set(xlabel = x_label, ylabel = z_label, title = title)
		ax.set_xlim([x1[0,0], x1[-1,0]])
		
		# Set colorbar
		cb = plt.colorbar()
		cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:]))

		# Show
		plt.show()

	def plot_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Plot a field at the level :math:`\theta = \bar{\theta}`.

		Args:
			field_to_plot (str): String specifying the field to plot. This may be 
				either a model variable or a diagnosed field, as:
				* 'horizontal_velocity', i.e., the horizontal velocity field.
			z_level (int): Index corresponding to the :math:`\theta`-level identifying 
				the :math:`(x,y)`-cross-section to plot.
			time_level (int): The index corresponding to the time level to plot.
			**kwargs: Keyword arguments defining various plotting settings.
		"""
		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, :, z_level, time_level]
		elif field_to_plot == 'horizontal_velocity':
			x, x_hl = self._grid.x.values, self._grid.x_half_levels.values
			y, y_hl = self._grid.y.values, self._grid.y_half_levels.values

			u = self._vars['x_velocity'].values[:, :, z_level, time_level]
			rbs_u = RectBivariateSpline(x_hl, y, u)
			interp_u = rbs_u(x, y)

			v = self._vars['y_velocity'].values[:, :, z_level, time_level]
			rbs_v = RectBivariateSpline(x, y_hl, v)
			interp_v = rbs_v(x, y)

			var = np.sqrt(interp_u ** 2 + interp_v ** 2)
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		nx, ny = self._grid.nx, self._grid.ny
		ni, nj = var.shape[0], var.shape[1]

		# Get keyword arguments
		title           = kwargs.get('title' , field_to_plot)
		x_label         = kwargs.get('x_label', '{} [${}$]'.format(self._grid.x.dims, self._grid.x.attrs.get('units', '')))
		x_factor        = kwargs.get('x_factor', 1.)
		y_label         = kwargs.get('y_label', '{} [${}$]'.format(self._grid.y.dims, self._grid.y.attrs.get('units', '')))
		y_factor        = kwargs.get('y_factor', 1.)
		cmap_name       = kwargs.get('cmap_name', 'RdYlBu')
		cmap_levels     = kwargs.get('cmap_levels', 31)
		cmap_center     = kwargs.get('cmap_center', None)
		cmap_half_width = kwargs.get('cmap_half_width', None)

		# The grid
		xv = x_factor * self._grid.x.values if ni == nx else x_factor * self._grid.x_half_levels.values
		yv = y_factor * self._grid.y.values if nj == ny else y_factor * self._grid.y_half_levels.values
		x, y = np.meshgrid(xv, yv, indexing = 'ij')

		# Instantiate figure and axis objects
		fig, ax = plt.subplots(figsize = [11,8])

		# Draw topography isolevels
		cs = plt.contour(x, y, self._vars['height'].values[:, :, -1, time_level], colors = 'gray')
		#plt.clabel(cs, inline = 1, fontsize = 10)

		# Create colormap
		field_min, field_max = np.amin(var), np.amax(var)
		if cmap_center is None or not (lt(field_min, cmap_center) and lt(cmap_center, field_max)):
			cmap_lb, cmap_ub = field_min, field_max
		else:
			half_width = max(cmap_center - field_min, field_max - cmap_center) if cmap_half_width is None else cmap_half_width
			cmap_lb, cmap_ub = cmap_center - half_width, cmap_center + half_width 
		color_scale = np.linspace(cmap_lb, cmap_ub, cmap_levels, endpoint = True)

		if cmap_name == 'BuRd':
			cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)

		# Plot the field
		plt.contourf(x, y, var, color_scale, cmap = cm)

		# Plot settings
		ax.set(xlabel = x_label, ylabel = y_label, title = title)
		ax.set_xlim([x_factor * self._grid.x.values[0], x_factor * self._grid.x.values[-1]])
		ax.set_ylim([y_factor * self._grid.y.values[0], y_factor * self._grid.y.values[-1]])
		
		# Set colorbar
		cb = plt.colorbar()
		cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:]))

		# Show
		plt.show()
