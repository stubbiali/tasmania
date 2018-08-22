def make_animation_contourf_xz(grid, states_list, field_to_plot, y_level, save_dest, **kwargs):
	"""
	Given a list of model states, generate an animation showing the time-evolution of the contourf plot 
	of a specified field at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	grid : obj
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	states_list : list
		List of model state dictionaries. Each state must necessarily contain either `height` 
		or `height_on_interface_levels`, and `time`.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state;
		* 'vertical_velocity', for the vertical velocity in a two-dimensional, steady-state, isentropic flow; \
			the model state should contain the following variables:

			- `air_isentropic_density`;
			- `x_momentum_isentropic`;
			- `height_on_interface_levels`.

	y_level : int 
		:obj:`j`-index identifying the cross-section.
	save_dest : str
		Path to the location where the movie will be saved. The path should include the format extension.
	**kwargs :
		Keyword arguments to specify different plotting settings. 
		See :func:`~tasmania.plot.contourf_xz.plot_animation_contourf_xz` for the complete list.

	Raise
	-----
	ValueError :
		If neither the grid, nor the model state, contains `height` or `height_on_interface_levels`.
	"""
	# Extract, compute, or interpolate the field to plot
	for t, state in enumerate(states_list):
		time_ = state['time']

		if field_to_plot in state.keys():
			var_ = state[field_to_plot].values[:, y_level, :]
		elif field_to_plot == 'vertical_velocity':
			assert grid.ny == 1, 'The input grid should consist of only one point in the y-direction.'
			assert y_level == 0, 'As the grid consists of only one point in the y-direction, y_level must be 0.'

			s, U, h = get_numpy_arrays(state, (slice(0,None), y_level, slice(0,None)), 
				'air_isentropic_density', 'x_momentum_isentropic', 'height_on_interface_levels')

			u = U / s
			h_mid  = 0.5 * (h[:, :-1] + h[:, 1:])
			h_mid_at_u_loc_ = 0.5 * (h_mid[:-1, :] + h_mid[1:, :])
			h_mid_at_u_loc  = np.concatenate((h_mid_at_u_loc_[0:1, :], h_mid_at_u_loc_, h_mid_at_u_loc_[-1:, :]), axis = 0)

			var_ = u * (h_mid_at_u_loc[1:, :] - h_mid_at_u_loc[:-1, :]) / grid.dx
		else:
			raise ValueError('Unknown field to plot {}.'.format(field_to_plot))

		if t == 0:
			time = [time_,]
			var  = np.copy(var_[:,:,np.newaxis])
		else:
			time.append(time_)
			var = np.concatenate((var, var_[:,:,np.newaxis]), axis = 2)

	# Shortcuts
	nx, nz = grid.nx, grid.nz
	ni, nk, nt = var.shape

	for t, state in enumerate(states_list):
		# The underlying x-grid
		x  = grid.x[:] if ni == nx else grid.x_at_u_locations[:]
		xv_ = np.repeat(x[:, np.newaxis], nk, axis = 1)

		# Extract the height of the main or interface levels
		try:
			z = get_numpy_arrays(state, (slice(0,None), y_level, slice(0,None)), 
									   ('height_on_interface_levels', 'height'))
		except KeyError:
			try:
				z = grid.height_on_interface_levels[:]
			except AttributeError:
				try:
					z = grid.height[:]
				except AttributeError:
					print("""Neither the grid, nor the state, contains either ''height'' 
							 or ''height_on_interface_levels''.""")

		# Reshape the extracted height of the vertical levels
		if z.shape[1] < nk:
			raise ValueError("""As the field to plot is vertically staggered, 
								''height_on_interface_levels'' is needed.""")
		_topo_ = z[:, -1]
		if z.shape[1] > nk:
			z = 0.5 * (z[:, :-1] + z[:, 1:])

		# The underlying z-grid
		zv_ = np.zeros((ni, nk), dtype = datatype)
		if ni == nx:
			zv_[:, :] = z[:, :]
		else:
			zv_[1:-1, :] = 0.5 * (z[:-1, :] + z[1:, :])
			zv_[0, :], zv_[-1, :] = zv_[1, :], zv_[-2, :]

		# The underlying topography
		if ni == nx:
			topo_= _topo_
		else:
			topo_ = np.zeros((nx + 1), dtype = datatype)
			topo_[1:-1] = 0.5 * (_topo_[:-1] + _topo_[1:])
			topo_[0], topo_[-1] = topo_[1], topo_[-2]

		if t == 0:
			xv   = np.copy(xv_[:,:,np.newaxis])
			zv   = np.copy(zv_[:,:,np.newaxis])
			topo = np.copy(topo_[:,np.newaxis])
		else:
			xv   = np.concatenate((xv, xv_[:,:,np.newaxis]), axis = 2)
			zv   = np.concatenate((zv, zv_[:,:,np.newaxis]), axis = 2)
			topo = np.concatenate((topo, topo_[:,np.newaxis]), axis = 1)

	# Plot
	plot_animation_contourf_xz(time, xv, zv, var, topo, save_dest, **kwargs)

def plot_animation_contourf_xz(time, x, z, field, topography, save_dest, **kwargs):
	"""
	Generate an animation showing the time evolution of the contourf plot of a gridded field 
	at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	time : list
		List of :class:`datetime.datetime`\s representing the time instant of each snapshot.
	x : array_like
		Three-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
		The first axis represents the :math:`x`-direction, the second axis represents the
		:math:`z`-direction, and the third axis represents the time dimension.
	z : array_like
		Three-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
		The first axis represents the :math:`x`-direction, the second axis represents the
		:math:`z`-direction, and the third axis represents the time dimension.
	field : array_like
		Three-dimensional :class:`numpy.ndarray` representing the field to plot.
		The first axis represents the :math:`x`-direction, the second axis represents the
		:math:`z`-direction, and the third axis represents the time dimension.
	topography : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying topography.
		The first axis represents the :math:`x`-direction and the second axis represents the time dimension.
	save_dest : str
		Path to the location where the movie will be saved. The path should include the format extension.
		
	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Specify the limits for the :math:`x`-axis, so that the portion of the :math:`x`-axis which is 
		visualized is :obj:`x_factor * (x_lim[0], x_lim[1])`. By default, the entire domain is shown.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Specify the limits for the :math:`z`-axis, so that the portion of the :math:`z`-axis which is 
		visualized is :obj:`z_factor * (z_lim[0], z_lim[1])`. By default, the entire domain is shown.
	field_bias : float
		Bias for the field, so that the contour lines for :obj:`field - field_bias` are drawn. Default is 0.
	field_factor : float
		Scaling factor for the field, so that the contour lines for :obj:`field_factor * field` are drawn. 
		If a bias is specified, then the contour lines for :obj:`field_factor * (field - field_bias)` are drawn.
		Default is 1.
	draw_z_isolines : bool
		:obj:`True` to draw the :math:`z`-isolines, :obj:`False` otherwise. Default is :obj:`True`.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	fps : int
		Frames per second. Default is 15.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	ni, nk, nt = field.shape

	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', 'z')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	field_bias       = kwargs.get('field_bias', 0.)
	field_factor     = kwargs.get('field_factor', 1.)
	draw_z_isolines  = kwargs.get('draw_z_isolines', True)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	fps				 = kwargs.get('fps', 15)
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	x_lim	   =  None if x_lim is None else [x_factor * lim for lim in x_lim] 
	z          *= z_factor
	z_lim	   =  None if z_lim is None else [z_factor * lim for lim in z_lim] 
	field      =  (field - field_bias) * field_factor
	topography *= z_factor

	# Instantiate figure and axes objects
	fig, ax = plt.subplots(figsize = figsize)

	# Determine color scale for colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = plot_utils.reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Instantiate writer class
	ffmpeg_writer = manimation.writers['ffmpeg']
	metadata = {'title': ''}
	writer = ffmpeg_writer(fps = fps, metadata = metadata)

	with writer.saving(fig, save_dest, nt):
		for n in range(nt):
			# Clean the canvas
			ax.cla()

			# Plot the z-isolines
			if draw_z_isolines:
				for k in range(nk):
					ax.plot(x[:, k, n], z[:, k, n], color = 'gray', linewidth = 1)

			# Plot the topography
			ax.plot(x[:, -1], topography[:, n], color = 'black', linewidth = 1)

			# Plot the field
			plt.contourf(x[:, :, n], z[:, :, n], field[:, :, n], color_scale, cmap = cm)

			# Set axes labels and figure title
			ax.set(xlabel = x_label, ylabel = z_label)
			plt.title(title, loc = 'left', fontsize = fontsize-1)

			# Set x-limits
			if x_lim is None:
				ax.set_xlim([x[0,0,n], x[-1,0,n]])
			else:
				ax.set_xlim(x_lim)

			# Set z-limits
			if z_lim is not None:
				ax.set_ylim(z_lim)

			# Add text
			if text is not None:
				ax.add_artist(AnchoredText(text, loc = text_loc))

			if n == 0:
				# Set colorbar
				cb = plt.colorbar(orientation = cbar_orientation)
				cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
				cb.ax.set_title(cbar_title)
				cb.ax.set_xlabel(cbar_x_label)
				cb.ax.set_ylabel(cbar_y_label)

			# Add time
			plt.title(str(convert_datetime64_to_datetime(time[n]) - convert_datetime64_to_datetime(time[0])), 
				      loc = 'right', fontsize = fontsize-1)

			# Set layout
			fig.tight_layout()

			# Let the writer grab the frame
			writer.grab_frame()

