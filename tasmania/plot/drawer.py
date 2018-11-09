"""
This module contains:
	Drawer
"""
import abc


class Drawer:
	"""
	This abstract base class represents a generic drawer.
	A *drawer* is a callable object which uses the data grabbed
	from an input state dictionary (or a time-series of state
	dictionaries) to generate a specific plot. The figure and
	the axes encapsulating the plot should be provided as well.

	Attributes
	----------
	settings : dict
		Dictionary specifying plot-specific settings.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, **kwargs):
		"""
		Parameters
		----------
		**kwargs :
			Keyword arguments specifying plot-specific settings.
		"""
		self.properties = kwargs

	@abc.abstractmethod
	def __call__(self, state, fig, ax):
		"""
		Call operator generating the plot.

		Parameters
		----------
		state : dict, sequence[dict]
			Either a state or a sequence of states from which
			retrieving the data used to draw the plot.
			A state is a dictionary whose keys are strings denoting
			model variables, and values are :class:`sympl.DataArray`\s
			storing values for those variables.
		fig : figure
			A :class:`matplotlib.pyplot.figure`.
		ax : axes
			An instance of :class:`matplotlib.axes.Axes` into which
			the plot will be encapsulated.
		"""
