from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt


def reverse_colormap(cmap, name=None):
	"""
	Reverse a Matplotlib colormap.

	Parameters
	----------
	cmap : obj 
		The :class:`matplotlib.colors.LinearSegmentedColormap` to invert.
	name : `str`, optional 
		The name of the reversed colormap. By default, this is obtained by appending '_r' 
		to the name of the input colormap.

	Return
	------
	obj :
		The reversed :class:`matplotlib.colors.LinearSegmentedColormap`.

	References
	----------
	https://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib.
	"""
	keys = []
	reverse = []

	for key in cmap._segmentdata:
		# Extract the channel
		keys.append(key)
		channel = cmap._segmentdata[key]

		# Reverse the channel
		data = []
		for t in channel:
			data.append((1-t[0], t[2], t[1]))
		reverse.append(sorted(data))

	# Set the name for the reversed map
	if name is None:
		name = cmap.name + '_r'

	return LinearSegmentedColormap(name, dict(zip(keys, reverse)))


def get_figure_and_axes(fig=None, ax=None, default_fig=None,
						figsize=None, fontsize=12, projection=None):
	"""
	Get a :class:`matplotlib.pyplot.figure` object and a :class:`matplotlib.axes.Axes` 
	object, with the latter embedded in the former. The returned values are determined 
	as follows.

	* If both :obj:`fig` and :obj:`ax` arguments are passed in:

		- if :obj:`ax` is embedded in :obj:`fig`, return :obj:`fig` and :obj:`ax`;
		- otherwise, return the figure which encloses :obj:`ax`, and :obj:`ax` itself.

	* If :obj:`fig` is provided but :obj:`ax` is not:

		- if :obj:`fig` contains some subplots, return :obj:`fig` and the axes of the
			first subplot it contains;
		- otherwise, add a subplot to :obj:`fig` in position (1,1,1) and return :obj:`fig`
			and the subplot axes.

	* If :obj:`ax` is provided but :obj:`fig` is not, return the figure which encloses 
		:obj:`ax`, and :obj:`ax` itself.

	* If neither :obj:`fig` nor :obj:`ax` are passed in:

		- if :obj:`default_fig` is not given, instantiate a new pair of figure and axes;
		- if :obj:`default_fig` is provided and it  contains some subplots, return 
			:obj:`default_fig` and the axes of the first subplot it contains;
		- if :obj:`default_fig` is provided but is does not contain any subplot, add a 
			subplot to :obj:`default_fig` in position (1,1,1) and return :obj:`default_fig`
			and the subplot axes.

	Parameters
	----------
	fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure` object.
	ax : `axes`, optional
		An instance of :class:`matplotlib.axes.Axes`.
	default_fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure` object.
	figsize : `tuple`, optional
		The size which the output figure should have.
		This argument is effective only if the figure is created within the function.
	fontsize : `int`, optional
		Font size for the output figure and axes. Default is 12.
		This argument is effective only if the figure is created within the function.
	projection : `str`, optional
		The axes projection.

	Returns
	-------
	out_fig : figure
		A :class:`matplotlib.pyplot.figure` object.
	out_ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	"""
	if (fig is not None) and (ax is not None):
		try:
			if ax not in fig.get_axes():
				import warnings
				warnings.warn("""Input axes do not belong to the input figure,
								 so consider the figure which the axes belong to.""",
							  RuntimeWarning)
				out_fig, out_ax = ax.get_figure(), ax
			else:
				out_fig, out_ax = fig, ax
		except AttributeError:
			import warnings
			warnings.warn("""Input argument ''fig'' does not seem to be a figure, 
							 so consider the figure the axes belong to.""", RuntimeWarning)
			out_fig, out_ax = ax.get_figure(), ax

	if (fig is not None) and (ax is None):
		try:
			out_fig = fig
			out_ax = fig.get_axes()[0] if len(fig.get_axes()) > 0 \
					 else fig.add_subplot(1, 1, 1, projection=projection)
		except AttributeError:
			import warnings
			warnings.warn("""Input argument ''fig'' does not seem to be a figure, 
							 hence a proper figure object is created.""", RuntimeWarning)
			rcParams['font.size'] = fontsize
			out_fig, out_ax = plt.subplots(figsize=figsize, projection=projection)

	if (fig is None) and (ax is not None):
		out_fig, out_ax = ax.get_figure(), ax

	if (fig is None) and (ax is None):
		if default_fig is None:
			rcParams['font.size'] = fontsize
			out_fig, out_ax = plt.subplots(figsize=figsize,
										   subplot_kw={'projection': projection})
		else:
			try:
				out_fig = default_fig
				out_ax = out_fig.get_axes()[0] if len(out_fig.get_axes()) > 0 \
						 else out_fig.add_subplot(1, 1, 1, projection=projection)
			except AttributeError:
				import warnings
				warnings.warn("""The argument ''default_fig'' does not actually seem 
								 to be a figure, hence a proper figure object is created.""",
							  RuntimeWarning)
				rcParams['font.size'] = fontsize
				out_fig, out_ax = plt.subplots(figsize=figsize,
											   subplot_kw={'projection': projection})

	return out_fig, out_ax


def set_plot_properties(ax, *, fontsize=12,
						title_center='', title_left='', title_right='',
						x_label='', x_lim=None, x_ticks=None,
						x_ticklabels=None, xaxis_visible=True,
						y_label='', y_lim=None, y_ticks=None,
						y_ticklabels=None, yaxis_visible=True,
						z_label='', z_lim=None, z_ticks=None,
						z_ticklabels=None, zaxis_visible=True,
						legend_on=False, legend_loc='best',
						text=None, text_loc='',
						grid_on=False):
	"""
	An utility to ease the configuration of two- and three-dimensional plots.

	Parameters
	----------
	ax : axes
		Instance of :class:`matplotlib.axes.Axes` enclosing the plot.
	fontsize : `int`, optional
		Font size to use for the plot titles, and axes ticks and labels.
		Default is 12.
	title_center : `str`, optional
		Text to use for the axes center title. Default is an empty string.
	title_left : `str`, optional
		Text to use for the axes left title. Default is an empty string.
	title_right : `str`, optional
		Text to use for the axes right title. Default is an empty string.
	x_label : `str`, optional
		Text to use for the label of the x-axis. Default is an empty string.
	x_lim : `tuple`, optional
		Data limits for the x-axis. Default is :obj:`None`, i.e., the data limits
		will be left unchanged.
	x_ticks : `list of float`, optional
		List of x-axis ticks location.
	x_ticklabels : `list of str`, optional
		List of x-axis ticks labels.
	xaxis_visible : `bool`, optional
		:obj:`False` to make the x-axis invisible. Default is :obj:`True`.
	y_label : `str`, optional
		Text to use for the label of the y-axis. Default is an empty string.
	y_lim : `tuple`, optional
		Data limits for the y-axis. Default is :obj:`None`, i.e., the data limits
		will be left unchanged.
	y_ticks : `list of float`, optional
		List of y-axis ticks location.
	y_ticklabels : `list of str`, optional
		List of y-axis ticks labels.
	yaxis_visible : `bool`, optional
		:obj:`False` to make the y-axis invisible. Default is :obj:`True`.
	z_label : `str`, optional
		Text to use for the label of the z-axis. Default is an empty string.
	z_lim : `tuple`, optional
		Data limits for the z-axis. Default is :obj:`None`, i.e., the data limits
		will be left unchanged.
	z_ticks : `list of float`, optional
		List of z-axis ticks location.
	z_ticklabels : `list of str`, optional
		List of z-axis ticks labels.
	zaxis_visible : `bool`, optional
		:obj:`False` to make the z-axis invisible. Default is :obj:`True`.
	legend_on : `bool`, optional
		:obj:`True` to show the legend, :obj:`False` otherwise. Default is :obj:`False`.
	legend_loc : `str`, optional
		String specifying the location where the legend box should be placed.
		Default is 'best'; please see :func:`matplotlib.pyplot.legend` for all
		the available options.
	text : str
		Text to be added to the figure as anchored text. Default is :obj:`None`,
		and no text box is shown.
	text_loc : str
		String specifying the location where the text box should be placed.
		Default is 'upper right'; please see :class:`matplotlib.offsetbox.AnchoredText`
		for all the available options.
	grid_on : `bool`, optional
		:obj:`True` to show the legend, :obj:`False` otherwise. Default is :obj:`False`.
	"""
	rcParams['font.size'] = fontsize

	if ax.get_title(loc='center') == '':
		ax.set_title(title_center, loc='center', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='left') == '':
		ax.set_title(title_left, loc='left', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='right') == '':
		ax.set_title(title_right, loc='right', fontsize=rcParams['font.size']-1)

	if ax.get_xlabel() == '':
		ax.set(xlabel=x_label)
	if ax.get_ylabel() == '':
		ax.set(ylabel=y_label)
	try:
		if ax.get_zlabel() == '':
			ax.set(zlabel=z_label)
	except AttributeError:
		if z_label != '':
			import warnings
			warnings.warn('The plot is not three-dimensional, therefore the '
					  	   'argument ''z_label'' is disregarded.', RuntimeWarning)
		else:
			pass

	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	try:
		if z_lim is not None:
			ax.set_zlim(z_lim)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_lim'' is disregarded.', RuntimeWarning)

	if x_ticks is not None:
		ax.get_xaxis().set_ticks(x_ticks)
	if y_ticks is not None:
		ax.get_yaxis().set_ticks(y_ticks)
	try:
		if z_ticks is not None:
			ax.get_zaxis().set_ticks(z_ticks)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_ticks'' is disregarded.', RuntimeWarning)

	if x_ticklabels is not None:
		ax.get_xaxis().set_ticklabels(x_ticklabels)
	if y_ticklabels is not None:
		ax.get_yaxis().set_ticklabels(y_ticklabels)
	try:
		if z_ticklabels is not None:
			ax.get_zaxis().set_ticklabels(z_ticklabels)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_ticklabels'' is disregarded.', RuntimeWarning)

	if not xaxis_visible:
		ax.get_xaxis().set_visible(False)
	if not yaxis_visible:
		ax.get_yaxis().set_visible(False)
	try:
		if not zaxis_visible:
			ax.get_zaxis().set_visible(False)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''zaxis_visible'' is disregarded.', RuntimeWarning)

	if legend_on:
		ax.legend(loc=legend_loc)

	if text is not None:
		ax.add_artist(AnchoredText(text, loc=text_loc))

	if grid_on:
		ax.grid(True)


def set_colorbar(fig, mappable, color_levels, *, cbar_ticks_step=1, cbar_ticks_pos='center',
				 cbar_title='', cbar_x_label='', cbar_y_label='',
				 cbar_orientation='vertical', cbar_ax=None):
	"""
	An utility to ease the configuration of the colorbar in Matplotlib plots.

	Parameters
	----------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	mappable : mappable
		The mappable, i.e., the image, which the colorbar applies
	color_levels : array_like
		1-D array of the levels corresponding to the colorbar colors.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the colorbar. Default is 1,
		i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the color bar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_x_label : str
		Label for the horizontal axis of the colorbar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the colorbar. Default is an empty string.
	cbar_title : str
		Title for the colorbar. Default is an empty string.
	cbar_orientation : str
		Orientation of the colorbar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the colorbar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the colorbar. If no indices are given,
		only the current axes are resized.
	"""
	if cbar_ax is None:
		cb = plt.colorbar(mappable, orientation=cbar_orientation)
	else:
		try:
			axes = fig.get_axes()
			cb = plt.colorbar(mappable, orientation=cbar_orientation,
                              ax=[axes[i] for i in cbar_ax])
		except TypeError:
			# cbar_ax is not iterable
			cb = plt.colorbar(mappable, orientation=cbar_orientation)
		except IndexError:
			# cbar_ax contains an index which exceeds the number of axes in the figure
			cb = plt.colorbar(mappable, orientation=cbar_orientation)

	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	if cbar_ticks_pos == 'center':
		cb.set_ticks(0.5 * (color_levels[:-1] + color_levels[1:])[::cbar_ticks_step])
	else:
		cb.set_ticks(color_levels[::cbar_ticks_step])
