from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
from sympl import Monitor

from tasmania.utils import plot_utils


class Plot1d(Monitor):
	"""
	A :class:`sympl.Monitor` visualizing a model state field along a line
	orthogonal to one of the coordinate planes.

	Attributes
	----------
    interactive : bool
        :obj:`True` to enable interactive plotting, :obj:`False` otherwise.
    fontsize : int
        Font size to use in the plot. Default is 16.
    figsize : tuple
        Tuple specifying the dimensions which the figure should have.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
    plot_properties : dict
        Plot properties which are forwarded to and thereby set by
        :func:`~tasmania.utils.plot_utils.set_plot_properties`.
    plot_function_kwargs : dict
        Plot settings which are forwarded to the plot function as keywords arguments.
	"""
	def __init__(self, grid, plot_function, field_to_plot, levels, interactive=True,
				 fontsize=16, figsize=(8, 8), tight_layout=True,
				 plot_properties=None, plot_function_kwargs=None):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`, or one of its derived classes.
		plot_function : function
			Pointer to the function which pulls out of the state any quantity
			required to generate the plot, and generates the plot itself.
			The signature of this function should be
			:obj:`fig, ax = plot_function(grid, state, field_to_plot, levels, fig, ax, **kwargs)`.
		field_to_plot : str
			String specifying the field to plot.
		levels : dict
			Dictionary whose keys are the ids of the two axes orthogonal to
			the cross line, and whose values are the corresponding indices
			identifying the cross line itself.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Default is :obj:`True`.
		fontsize : `int`, optional
			Font size to use in the plot. Default is 16.
		figsize : `tuple`, optional
			Tuple specifying the dimensions which the figure should have. Default is (8, 8).
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		plot_properties : `dict`, optional
			Plot properties which are forwarded and thereby set by
			:func:`~tasmania.utils.plot_utils.set_plot_properties`.
			Default is :obj:`None`.
		plot_function_kwargs : `dict`, optional
			Plot settings which are forwarded to the plot function as keywords arguments.
			Default is :obj:`None`.
		"""
		self._assert_plot_function_signature(plot_function)

		self._grid			= grid
		self._plot_function = plot_function
		self._field_to_plot = field_to_plot
		self._levels        = levels

		self.interactive 		  = interactive
		self.fontsize			  = fontsize
		self.figsize			  = figsize
		self.tight_layout		  = tight_layout
		self.plot_properties 	  = {} if not isinstance(plot_properties, dict) \
									else plot_properties
		self.plot_function_kwargs = {} if not isinstance(plot_function_kwargs, dict) \
									else plot_function_kwargs

		self._figure = None

	def store(self, state, fig=None, ax=None, save_dest=None, show=False):
		"""
		Update the plot using the given state.

		Parameters
		----------
		state : dict
			A model state dictionary.
		fig : `figure`, optional
			A :class:`matplotlib.pyplot.figure`.
		ax : `axes`, optional
			An instance of :class:`matplotlib.axes.Axes`. 
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on, :obj:`True` to show the figure, 
			:obj:`False` otherwise. Default is :obj:`False`.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		out_ax : axes
			The :class:`matplotlib.axes.Axes` object enclosing the plot.
		"""
		self._set_figure(fig)

		out_fig, out_ax = plot_utils.get_figure_and_axes(
			fig, ax, default_fig=self._figure, figsize=self.figsize, fontsize=self.fontsize)
			
		if ax is None:
			out_ax.cla()

		out_fig, out_ax = self._plot_function(self._grid, state,
											  self._field_to_plot, self._levels,
			 					   	  		  out_fig, out_ax, **self.plot_function_kwargs)

		plot_utils.set_plot_properties(out_ax, **self.plot_properties)

		if fig is None and self.tight_layout:
			out_fig.tight_layout()

		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		if not self.interactive and show:
			plt.show()

		return out_fig, out_ax

	@staticmethod
	def _assert_plot_function_signature(plot_function):
		try:
			from inspect import getfullargspec as getargspec	
		except ImportError:
			from inspect import getargspec

		argspec = getargspec(plot_function)
		assert argspec.args == ['grid', 'state', 'field_to_plot', 'levels', 'fig', 'ax'], \
			"""The signature of the plot function should be 
			   fig, ax = plot_function(grid, state, field_to_plot, levels, fig, ax, **kwargs)."""

	def _set_figure(self, fig=None):
		if fig is not None:
			self._figure = None
			return

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = self.fontsize
				self._figure = plt.figure(figsize=self.figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = self.fontsize
			self._figure = plt.figure(figsize=self.figsize)


class Plot2d(Monitor):
	"""
	A :class:`sympl.Monitor` visualizing a model state field in a cross section
	parallel to one of the coordinate planes.

	Attributes
	----------
    interactive : bool
        :obj:`True` to enable interactive plotting, :obj:`False` otherwise.
    fontsize : int
        Font size to use in the plot. Default is 16.
    figsize : tuple
        Tuple specifying the dimensions which the figure should have.
    projection : `str`, optional
        Projection for the plot axes.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
    plot_properties : dict
        Plot properties which are forwarded to and thereby set by
        :func:`~tasmania.utils.plot_utils.set_plot_properties`.
    plot_function_kwargs : dict
        Plot settings which are forwarded to the plot function as keywords arguments.
	"""
	def __init__(self, grid, plot_function, field_to_plot, level, interactive=True,
				 fontsize=16, figsize=(8, 8), projection=None, tight_layout=True,
				 plot_properties=None, plot_function_kwargs=None):
		"""
		The constructor.

		Parameters
		----------
		plot_function : function
			Pointer to the function which pulls out of the state any quantity
			required to generate the plot, and generates the plot itself.
			The signature of this function should be either
			:obj:`fig = plot_function(grid, state, field_to_plot, x_level, fig, ax, **kwargs)`,
			:obj:`fig = plot_function(grid, state, field_to_plot, y_level, fig, ax, **kwargs)`,
			or
			:obj:`fig = plot_function(grid, state, field_to_plot, z_level, fig, ax, **kwargs)`.
		grid : grid
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes.
		field_to_plot : str
			String specifying the field to plot.
		level : int
			Either the :obj:`i`-index identifying the :math:`yz`-cross-section,
			the :obj:`j`-index identifying the :math:`xz`-cross-section, or
			the :obj:`k`-index identifying the :math:`xy`-cross-section.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Default is :obj:`True`.
		fontsize : `int`, optional
			Font size to use in the plot. Default is 16.
		figsize : `tuple`, optional
			Tuple specifying the dimensions which the figure should have. Default is (8, 8).
		projection : `str`, optional
			Projection for the plot axes. Default is :obj:`None`, i.e., the standard
			Cartesian projection is used.
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		plot_properties : `dict`, optional
			Plot properties which are forwarded to and thereby set by
			:func:`~tasmania.utils.plot_utils.set_plot_properties`.
			Default is :obj:`None`.
		plot_function_kwargs : `dict`, optional
			Plot settings which are forwarded to the plot function as keywords arguments.
			Default is :obj:`None`.
		"""
		self._assert_plot_function_signature(plot_function)

		self._grid 			= grid
		self._plot_function = plot_function
		self._field_to_plot = field_to_plot
		self._level 		= level

		self.interactive 		  = interactive
		self.fontsize			  = fontsize
		self.figsize			  = figsize
		self.projection			  = projection
		self.tight_layout		  = tight_layout
		self.plot_properties 	  = {} if not isinstance(plot_properties, dict) \
									else plot_properties
		self.plot_function_kwargs = {} if not isinstance(plot_function_kwargs, dict) \
									else plot_function_kwargs

		self._figure = None

	def store(self, state, fig=None, ax=None, save_dest=None, show=False):
		"""
		Update the plot using the given state.

		Parameters
		----------
		state : dict
			A model state dictionary.
		fig : figure
			A :class:`matplotlib.pyplot.figure`.		
		ax : axes		
			An instance of :class:`matplotlib.axes.Axes`.
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on, :obj:`True` to show the figure, 
			:obj:`False` otherwise. Default is :obj:`False`.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		out_ax : axes
			The :class:`matplotlib.axes.Axes` object enclosing the plot.
		"""
		self._set_figure(fig)

		out_fig, out_ax = plot_utils.get_figure_and_axes(
			fig, ax, default_fig=self._figure, figsize=self.figsize,
			fontsize=self.fontsize, projection=self.projection)

		if ax is None:
			out_ax.cla()

		out_fig, out_ax = self._plot_function(self._grid, state,
											  self._field_to_plot, self._level,
											  out_fig, out_ax, **self.plot_function_kwargs)

		plot_utils.set_plot_properties(out_ax, **self.plot_properties)

		if fig is None and self.tight_layout:
			out_fig.tight_layout()

		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		if not self.interactive and show:
			plt.show()

		return out_fig, out_ax

	@staticmethod
	def _assert_plot_function_signature(plot_function):
		try:
			from inspect import getfullargspec as getargspec
		except ImportError:
			from inspect import getargspec

		argspec = getargspec(plot_function)
		assert (argspec.args == ['grid', 'state', 'field_to_plot', 'x_level', 'fig', 'ax'] or
				argspec.args == ['grid', 'state', 'field_to_plot', 'y_level', 'fig', 'ax'] or
				argspec.args == ['grid', 'state', 'field_to_plot', 'z_level', 'fig', 'ax']), \
			"""The signature of the plot function should be either
			   fig = plot_function(grid, state, field_to_plot, x_level, fig, ax, **kwargs),
			   fig = plot_function(grid, state, field_to_plot, y_level, fig, ax, **kwargs), or
			   fig = plot_function(grid, state, field_to_plot, z_level, fig, ax, **kwargs)."""

	def _set_figure(self, fig=None):
		if fig is not None:
			self._figure = None
			return

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = self.fontsize
				self._figure = plt.figure(figsize=self.figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = self.fontsize
			self._figure = plt.figure(figsize=self.figsize)


class Plot3d(Monitor):
	"""
	A :class:`sympl.Monitor` visualizing a three-dimensional field.
	This class may be used, e.g., to plot the underlying topography.

	Attributes
	----------
    interactive : bool
        :obj:`True` to enable interactive plotting, :obj:`False` otherwise.
    fontsize : int
        Font size to use in the plot. Default is 16.
    figsize : tuple
        Tuple specifying the dimensions which the figure should have.
    projection : `str`, optional
        Projection for the plot axes.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
    plot_properties : dict
        Plot properties which are forwarded to and thereby set by
        :func:`~tasmania.utils.plot_utils.set_plot_properties`.
    plot_function_kwargs : dict
        Plot settings which are forwarded to the plot function as keywords arguments.
	"""
	def __init__(self, grid, plot_function, field_to_plot, interactive=True,
				 fontsize=16, figsize=(8, 8), projection='3d', tight_layout=True,
				 plot_properties=None, plot_function_kwargs=None):
		"""
		The constructor.

		Parameters
		----------
		plot_function : function
			Pointer to the function which pulls out of the state any quantity required
			to generate the plot, and generates the plot itself. The signature of this
			function should be either
			:obj:`fig = plot_function(grid, state, field_to_plot, y_level, fig, ax, **kwargs)`
			or
			:obj:`fig = plot_function(grid, state, field_to_plot, z_level, fig, ax, **kwargs)`.
		grid : grid
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes.
		field_to_plot : str
			String specifying the field to plot.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Default is :obj:`True`.
		fontsize : `int`, optional
			Font size to use in the plot. Default is 16.
		figsize : `tuple`, optional
			Tuple specifying the dimensions which the figure should have. Default is (8, 8).
		projection : `str`, optional
			Projection for the plot axes. Default is '3d'.
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		plot_properties : `dict`, optional
			Plot properties which are forwarded to and thereby set by
			:func:`~tasmania.utils.plot_utils.set_plot_properties`.
			Default is :obj:`None`.
		plot_function_kwargs : `dict`, optional
			Plot settings which are forwarded to the plot function as keywords arguments.
			Default is :obj:`None`.
		"""
		self._assert_plot_function_signature(plot_function)

		self._grid 			= grid
		self._plot_function = plot_function
		self._field_to_plot = field_to_plot

		self.interactive 		  = interactive
		self.fontsize			  = fontsize
		self.figsize			  = figsize
		self.projection			  = projection
		self.tight_layout		  = tight_layout
		self.plot_properties 	  = {} if not isinstance(plot_properties, dict) \
									else plot_properties
		self.plot_function_kwargs = {} if not isinstance(plot_function_kwargs, dict) \
									else plot_function_kwargs

		self._figure = None

	def store(self, state, fig=None, ax=None, save_dest=None, show=False):
		"""
		Update the plot using the given state.

		Parameters
		----------
		state : dict
			A model state dictionary.
		fig : figure
			A :class:`matplotlib.pyplot.figure`.
		ax : axes
			An instance of :class:`matplotlib.axes.Axes`.
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on, :obj:`True` to show the figure,
			:obj:`False` otherwise. Default is :obj:`False`.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		out_ax : axes
			The :class:`matplotlib.axes.Axes` object enclosing the plot.
		"""
		self._set_figure(fig)

		out_fig, out_ax = plot_utils.get_figure_and_axes(
			fig, ax, default_fig=self._figure, figsize=self.figsize,
			fontsize=self.fontsize, projection=self.projection)

		if ax is None:
			out_ax.cla()

		out_fig, out_ax = self._plot_function(self._grid, state, self._field_to_plot,
											  out_fig, out_ax, **self.plot_function_kwargs)

		plot_utils.set_plot_properties(out_ax, **self.plot_properties)

		if fig is None and self.tight_layout:
			out_fig.tight_layout()

		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		if not self.interactive and show:
			plt.show()

		return out_fig, out_ax

	@staticmethod
	def _assert_plot_function_signature(plot_function):
		try:
			from inspect import getfullargspec as getargspec
		except ImportError:
			from inspect import getargspec

		argspec = getargspec(plot_function)
		assert argspec.args == ['grid', 'state', 'field_to_plot', 'fig', 'ax'], \
			"""The signature of the plot function should be
			   fig = plot_function(grid, state, field_to_plot, fig, ax, **kwargs)."""

	def _set_figure(self, fig=None):
		if fig is not None:
			self._figure = None
			return

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = self.fontsize
				self._figure = plt.figure(figsize=self.figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = self.fontsize
			self._figure = plt.figure(figsize=self.figsize)
