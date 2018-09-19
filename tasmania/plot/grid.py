"""
This module contains:
	plot_grid_xz
	plot_grid_yz
	plot_grid_vertical_section
"""
import matplotlib as mpl
import numpy as np

from tasmania.utils.data_utils import get_numpy_arrays


def plot_grid_xz(grid, state, field_to_plot, y_level, fig, ax, **kwargs):
	"""
	Function which visualizes a :math:`xz`-cross-section of a grid.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str
		String specifying the field to plot. This is actually a dummy parameter,
		since it is not used within the function. Nevertheless, it is retained in
		the function signature for compliancy with
		:class:`~tasmania.plot.plot_monitors.Plot2d`.
	y_level : int
		:math:`y`-index identifying the cross-section.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.grid.plot_grid_vertical_section` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Raises
	------
	ValueError :
		If neither the grid, nor the model state, contains `height` nor
		`height_on_interface_levels`.
	"""
	# Extract the geometric height at the main or interface levels, and the topography
	try:
		zv = get_numpy_arrays(state, (slice(0, None), y_level, slice(0, None)),
							 'height_on_interface_levels')
		topo = np.copy(zv[:, -1])
	except KeyError:
		try:
			zv = grid.height_on_interface_levels.values
			topo = np.copy(zv[:, -1])
		except AttributeError:
			try:
				zv = get_numpy_arrays(state, (slice(0, None), y_level, slice(0, None)),
                                      'height')
				topo = grid.topography_height
			except KeyError:
				try:
					zv = grid.height.values
					topo = grid.topography_height
				except AttributeError:
					print("""Neither the grid, nor the state, contains either ''height'' 
                             or ''height_on_interface_levels''.""")

	# The underlying x-grid
	ni, nk = zv.shape
	x  = grid.x.values[:] if ni == grid.nx else grid.x_at_u_locations.values[:]
	xv = np.repeat(x[:, np.newaxis], nk, axis=1)

	# Plot and return figure object
	return plot_grid_vertical_section(xv, zv, topo, fig, ax, **kwargs)


def plot_grid_yz(grid, state, field_to_plot, x_level, fig, ax, **kwargs):
	"""
	Function which visualizes a :math:`yz`-cross-section of a grid.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str
		String specifying the field to plot. This is actually a dummy parameter,
		since it is not used within the function. Nevertheless, it is retained in
		the function signature for compliancy with :class:`~tasmania.plot.plot_monitors.Plot2d`.
	x_level : int
		:math:`x`-index identifying the cross-section.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.grid.plot_grid_vertical_section` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Raises
	------
	ValueError :
		If neither the grid, nor the model state, contains `height` nor
		`height_on_interface_levels`.
	"""
	# Extract the geometric height at the main or interface levels, and the topography
	try:
		zv = get_numpy_arrays(state, (x_level, slice(0, None), slice(0, None)),
							  'height_on_interface_levels')
		topo = np.copy(zv[:, -1])
	except KeyError:
		try:
			zv = grid.height_on_interface_levels.values
			topo = np.copy(zv[:, -1])
		except AttributeError:
			try:
				zv = get_numpy_arrays(state, (x_level, slice(0, None), slice(0, None)),
									  'height')
				topo = grid.topography_height
			except KeyError:
				try:
					zv = grid.height.values
					topo = grid.topography_height
				except AttributeError:
					print("""Neither the grid, nor the state, contains either ''height'' 
                             or ''height_on_interface_levels''.""")

	# The underlying x-grid
	nj, nk = zv.shape
	y  = grid.y.values[:] if nj == grid.ny else grid.y_at_v_locations.values[:]
	yv = np.repeat(y[:, np.newaxis], nk, axis=1)

	# Plot and return figure object
	return plot_grid_vertical_section(yv, zv, topo, fig, ax, **kwargs)


def plot_grid_vertical_section(hor_grid, vert_grid, topography, fig, ax, **kwargs):
	"""
	Function which visualizes a vertical cross-section of a grid.

	Parameters
	----------
	hor_grid : array_like
		2-D :class:`numpy.ndarray` representing the horizontal coordinates
		of the grid points.
	vert_grid : array_like
		2-D :class:`numpy.ndarray` representing the vertical coordinates
		of the grid points.
	topography : array_like
		1-D :class:`numpy.ndarray` representing the topography.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used in the plot. Default is 16.
	hor_factor : float
		Scaling factor for the horizontal axis. Default is 1.
	vert_factor : float
		Scaling factor for the vertical axis. Default is 1.
	linecolor : str
		String specifying the line color. Default is 'blue'.
	linestyle : str
		String specifying the line style. The default line style is '-'.
	linewidth : float
		The line width. Default is 1.5.
	fill_color : str
		The color used to fill the area below the topography profile.
		Default is 'white'.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Shortcuts
	ni, nk = hor_grid.shape

	# Get keyword arguments
	fontsize     = kwargs.get('fontsize', 16)
	hor_factor   = kwargs.get('hor_factor', 1.)
	vert_factor  = kwargs.get('vert_factor', 1.)
	linecolor	 = kwargs.get('linecolor', 'blue')
	linestyle    = kwargs.get('linestyle', '-')
	linewidth    = kwargs.get('linewidth', 1.5)
	fill_color   = kwargs.get('fill_color', 'white')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	hor_grid   *= hor_factor
	vert_grid  *= vert_factor
	topography *= vert_factor

	# Plot the grid
	for k in range(nk):
		ax.plot(hor_grid[:, k], vert_grid[:, k],
				color=linecolor, linestyle=linestyle, linewidth=linewidth)

	# Plot the topography
	ax.plot(hor_grid[:, -1], topography, color='black', linewidth=linewidth+0.5)

	# Fill the area under the topography
	ax.fill_between(hor_grid[:, -1], 0, topography, color=fill_color)

	# Bring the axes and the topography back to the original dimensions
	hor_grid   /= hor_factor
	vert_grid  /= vert_factor
	topography /= vert_factor

	return fig, ax
