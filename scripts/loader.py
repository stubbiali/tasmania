import abc
import tasmania as taz


class Loader:
	def __init__(self, filename):
		self._fname = filename
		self._grid, self._states = taz.load_netcdf_dataset(filename)

	@property
	def nt(self):
		return len(self._states)

	def get_state(self, tlevel):
		state = self._states[tlevel]
		self._grid.update_topography(state['time'] - self._states[0]['time'])
		return state

	def get_grid(self):
		return self._grid


class LoaderComposite:
	def __init__(self, loaders):
		from tasmania.utils.utils import assert_sequence
		assert_sequence(loaders, reftype=(Loader, LoaderComposite))

		self._loaders = loaders

	def get_state(self, tlevel):
		return_list = []

		for loader in self._loaders:
			return_list.append(loader.get_state(tlevel))

		return return_list

	def get_grid(self):
		return_list = []

		for loader in self._loaders:
			return_list.append(loader.get_grid())

		return return_list


class LoaderFactory:
	ledger = {}

	__metaclass__ = abc.ABCMeta

	@staticmethod
	def factory(filename):
		try:
			return LoaderFactory.ledger[filename]
		except KeyError:
			loader = Loader(filename)
			LoaderFactory.ledger[filename] = loader
			return loader
