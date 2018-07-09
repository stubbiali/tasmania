import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tasmania.namelist import datatype
import tasmania.plot.utils as plot_utils
from tasmania.utils.utils import get_numpy_arrays, equal_to as eq, \
								 smaller_or_equal_than as lt


def make_contourf_xy(grid, state, field_to_plot, z_level, fig, ax, **kwargs):
	"""
	Given an input model state, generate the contourf plot of a specified field at a cross-section
	parallel to the :math:`xy`-plane.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state;
		* 'horizontal_velocity', for the horizontal velocity; \
			the current object must contain either:
		
			- `x_velocity` and `y_velocity`;
			- `x_velocity_at_u_locations` and `y_velocity_at_v_locations`;
			- `air_density`, `x_momentum`, and `y_momentum`;
			- `air_isentropic_density`, `x_momentum_isentropic`, and `y_momentum_isentropic`.

	z_level : int 
		:math:`z`-index identifying the cross-section.
    fig : figure
        A :class:`matplotlib.pyplot.figure`.
    ax : axes
        An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.contourf_xy.plot_contourf_xy` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Extract, compute, or interpolate the field to plot
	if field_to_plot in state.keys():
		var = state[field_to_plot].values[:, :, z_level]
	elif field_to_plot == 'horizontal_velocity':
		try:
			u, v = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'x_velocity', 'y_velocity')
		except KeyError:
			pass

		try:
			u, v = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'x_velocity_at_u_locations', 'y_velocity_at_v_locations')
			u = 0.5 * (u[:-1, :] + u[1:, :])
			v = 0.5 * (v[:, :-1] + v[:, 1:])
		except KeyError:
			pass

		try:
			rho, ru, rv = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'air_density', 'x_momentum', 'y_momentum')
			u, v = ru / rho, rv / rho
		except KeyError:
			pass

		try:
			s, su, sv = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'air_isentropic_density', 'x_momentum_isentropic', 'y_momentum_isentropic')
			u, v = su / s, sv / s
		except KeyError:
			pass

		var = np.sqrt(u ** 2 + v ** 2)
	else:
		raise RuntimeError('Unknown field to plot.')

	# Shortcuts
	nx, ny = grid.nx, grid.ny
	ni, nj = var.shape

	# The underlying x-grid
	x  = grid.x[:] if ni == nx else grid.x_at_u_locations[:]
	xv = np.repeat(x[:, np.newaxis], nj, axis=1)

	# The underlying y-grid
	y  = grid.y[:] if nj == ny else grid.y_at_v_locations[:]
	yv = np.repeat(y[np.newaxis, :], ni, axis=0)

	# The topography height
	topo_ = np.copy(grid.topography_height)
	topo  = np.zeros((ni, nj), dtype=datatype) 
	if ni == nx and nj == ny:
		topo[:, :] = topo_[:, :]
	elif ni == nx + 1 and nj == ny:
		topo[1:-1, :] = 0.5 * (topo_[:-1, :] + topo_[1:, :])
		topo[0, :], topo[-1, :] = topo[1, :], topo[-2, :]
	elif ni == nx and nj == ny + 1:
		topo[:, 1:-1] = 0.5 * (topo_[:, :-1] + topo_[:, 1:])
		topo[:, 0], topo[:, -1] = topo[:, 1], topo[:, -2]
	else:
		topo[1:-1, 1:-1] = 0.25 * (topo_[:-1, :-1] + topo_[1:, :-1] +
								  topo_[:-1, :1]  + topo_[1:, 1:])
		topo[0, 1:-1], topo[-1, 1:-1] = topo[1, 1:-1], topo[-2, 1:-1]
		topo[:, 0], topo[:, -1] = topo[:, 1], topo[:, -2]

	# Plot
	return plot_contourf_xy(xv, yv, var, topo, fig, ax, **kwargs)


def plot_contourf_xy(x, y, field, topography, fig, ax, **kwargs):
	"""
	Generate the contourf plot of a field at a cross-section parallel to the :math:`xy`-plane.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` representing the :math:`x`-coordinates
		of the grid points.
	y : array_like
		2-D :class:`numpy.ndarray` representing the :math:`y`-coordinates
		of the grid points.
	field : array_like
		2-D :class:`numpy.ndarray` representing the field to plot.
	topography : array_like
		2-D :class:`numpy.ndarray` representing the underlying topography height.
    fig : figure
        A :class:`matplotlib.pyplot.figure`.
    ax : axes
        An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 16.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	y_factor : float
		Scaling factor for the :math:`y`-axis. Default is 1.
	field_bias : float
		Bias for the field, so that the contourf plot for :obj:`field - field_bias`
		is generated. Default is 0.
	field_factor : float
		Scaling factor for the field, so that the contourf plot for
		:obj:`field_factor * field` is generated. If a bias is specified, then the
		contourf plot for :obj:`field_factor * (field - field_bias)` is generated.
		Default is 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
	cbar_on : bool
		:obj:`True` to show the color bar, :obj:`False` otherwise. Default is :obj:`True`.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1,
		i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the color bar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers
		the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar
		covers the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_title : str
		Title for the color bar. Default is an empty string.
	cbar_orientation : str
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the color bar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the color bar. If no indices are given,
		only the current axes are resized.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	x_factor         = kwargs.get('x_factor', 1.)
	y_factor         = kwargs.get('y_factor', 1.)
	field_bias		 = kwargs.get('field_bias', 0.)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_on			 = kwargs.get('cbar_on', True)
	cbar_levels		 = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_ticks_pos	 = kwargs.get('cbar_ticks_pos', 'center')
	cbar_center		 = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label	 = kwargs.get('cbar_x_label', '')
	cbar_y_label	 = kwargs.get('cbar_y_label', '')
	cbar_title		 = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	cbar_ax			 = kwargs.get('cbar_ax', None)

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x     *= x_factor
	y     *= y_factor
	field -= field_bias
	field *= field_factor

	# Draw topography isolevels
	plt.contour(x, y, topography, colors='gray')

	# Create color bar for colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center-field_min, field_max-cbar_center) \
					 if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center-half_width, cbar_center+half_width 
	color_levels = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint=True)
	if eq(color_levels[0], color_levels[-1]):
		color_levels = np.linspace(cbar_lb-1e-8, cbar_ub+1e-8, cbar_levels, endpoint=True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = plot_utils.reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the field
	surf = plt.contourf(x, y, field, color_levels, cmap=cm)

	# Set the color bar
	if cbar_on:
		plot_utils.set_colorbar(fig, surf, color_levels,
								cbar_ticks_step=cbar_ticks_step,
								cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
								cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
								cbar_orientation=cbar_orientation, cbar_ax=cbar_ax)

	# Bring axes and field back to original units
	x     /= x_factor
	y 	  /= y_factor
	field /= field_factor
	field += field_bias

	return fig, ax
