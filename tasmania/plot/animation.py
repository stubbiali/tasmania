"""
This module contains:
	Animation
"""
import matplotlib.animation as manimation

from tasmania.plot.monitors import Plot, PlotComposite


class Animation:
	"""
	This class creates an animation by leveraging a wrapped
	:class:`~tasmania.plot.monitors.Plot` or
	:class:`~tasmania.plot.monitors.PlotComposite`
	to generate the frames.
	"""
	def __init__(self, artist, print_time=None, fps=15):
		"""
		The constructor.

		Parameters
		----------
		artist : artist
			Instance of :class:`~tasmania.plot.monitors.Plot` or
			:class:`~tasmania.plot.monitors.PlotComposite`.
		print_time : str
			String specifying if time should be printed above the plot,
			flush with the right edge. Options are:

				* 'elapsed', to print the time elapsed from the first snapshot stored;
				* 'absolute', to print the absolute time of the snapshot.
				* anything else, not to print anything.

			Default is :obj:`None`.
		fps : int
			Frames per second. Default is 15.
		"""
		# Store input arguments as private attributes
		self._artist = artist
		self._print_time = print_time
		self._fps = fps

		# Ensure the artist is in non-interactive mode
		self._artist.interactive = False

		# Initialize the list of states
		self._states = []

	def store(self, states):
		"""
		Append a new state (respectively, a list of states), to the list of
		states (resp., lists of states) stored in this object.

		Parameters
		----------
		states : dict or list
			A model state dictionary, or a list of model state dictionaries.
		"""
		self._states.append(states)

	def reset(self):
		"""
		Empty the list of stored states.
		"""
		self._states = []

	def run(self, save_dest):
		"""
		Generate the animation based on the list of states stored in this object.

		Parameters
		----------
		save_dest : str
			Path to the location where the movie should be saved.
			The path should include the format extension.
		"""
		nt = len(self._states)
		if nt == 0:
			import warnings
			warnings.warn('This object does not contain any model state, '
						  'so no movie will be created.')
			return

		# Instantiate writer class
		ffmpeg_writer = manimation.writers['ffmpeg']
		metadata = {'title': ''}
		writer = ffmpeg_writer(fps=self._fps, metadata=metadata)

		# Retrieve the figure object from the artist
		fig = self._artist.figure

		# Save initial time
		try:
			init_time = self._states[0]['time']
		except TypeError:
			init_time = self._states[0][0]['time']

		with writer.saving(fig, save_dest, nt):
			for n in range(nt):
				# Clean the canvas
				fig.clear()

				# Get current time
				try:
					time = self._states[n]['time']
				except TypeError:
					time = self._states[n][0]['time']

				# Get the string with the time
				if self._print_time == 'elapsed':
					time_str = str(time - init_time)
				elif self._print_time == 'absolute':
					time_str = str(time)
				else:
					time_str = None

				# Update artist(s)'s properties
				if time_str is not None:
					if isinstance(self._artist, PlotComposite):
						for subplot_artist in self._artist.artists:
							subplot_artist.axes_properties['title_right'] = time_str
					else:
						self._artist.axes_properties['title_right'] = time_str

				# Create the frame
				fig, _ = self._artist.store(self._states[n], fig=fig, show=False)

				# Let the writer grab the frame
				writer.grab_frame()
