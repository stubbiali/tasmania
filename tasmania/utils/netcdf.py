import netCDF4 as nc4
import numpy as np
import sympl


class NetCDFMonitor(sympl.NetCDFMonitor):
	"""
	Customized version of :class:`sympl.NetCDFMonitor`, which
	caches stored states and then write them to a NetCDF file,
	together with some grid properties.
	"""
	def __init__(self, filename, grid, time_units='seconds',
				 store_names=None, write_on_store=False, aliases=None):
		"""
		The constructor.

		Parameters
		----------
		filename : str
            The file to which the NetCDF file will be written.
        grid : grid
        	The underlying computational mesh, as an instance
        	of :class:`~tasmania.dynamics.grids.grid_xyz.GridXYZ`.
        time_units : str, optional
			The units in which time will be
			stored in the NetCDF file. Time is stored as an integer
			number of these units. Default is seconds.
        store_names : iterable of str, optional
        	Names of quantities to store. If not given,
        	all quantities are stored.
		write_on_store : bool, optional
	        If True, stored changes are immediately written to file.
			This can result in many file open/close operations.
			Default is to write only when the write() method is
            called directly.
        aliases : dict
			A dictionary of string replacements to apply to state variable
            names before saving them in netCDF files.
		"""
		super().__init__(filename, time_units, store_names, write_on_store,
						 aliases)
		self._grid = grid

	def write(self):
		"""
		Write grid properties and all cached states to the NetCDF file,
		and clear the cache. This will append to any existing NetCDF file.
		"""
		super().write()

		with nc4.Dataset(self._filename, self._write_mode) as dataset:
			topo = self._grid.topography

			dataset.createDimension('str_dim', 1)
			topo_type    = dataset.createVariable('topo_type', str, ('str_dim',))
			topo_type[:] = np.array([topo.topo_type], dtype='object')

			dataset.createDimension('scalar_dim', 1)
			topo_time 	 = dataset.createVariable('topo_time', float, ('scalar_dim',))
			topo_time[:] = np.array([topo.topo_time.total_seconds()], dtype=float)
			topo_time.setncattr('units', 's')
