import abc
import tasmania as taz


class Loader:
	def __init__(self, filename):
		self._fname = filename
		self._grid, self._states = taz.load_netcdf_dataset(filename)

	def __call__(self, tlevel):
		state = self._states[tlevel]
		self._grid.update_topography(state['time'] - self._states[0]['time'])
		return state


class LoaderComposite:
	def __init__(self, loaders):
		from tasmania.utils.utils import assert_sequence
		assert_sequence(loaders, reftype=(Loader, LoaderComposite))

		self._loaders = loaders

	def __call__(self, tlevel):
		return_list = []

		for loader in self._loaders:
			return_list.append(loader(tlevel))

		return return_list


class LoaderFactory:
	ledger = {}

	__metaclass__ = abc.ABCMeta

	@staticmethod
	def __call__(filename):
		try:
			return LoaderFactory.ledger[filename]
		except KeyError:
			loader = Loader(filename)
			LoaderFactory.ledger[filename] = loader
			return loader
