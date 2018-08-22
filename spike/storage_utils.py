def convert_variable_name(name):
	if name == 'isentropic_density':
		return 'air_isentropic_density'
	elif name == 'pressure' or name == 'air_pressure':
		return 'air_pressure_on_interface_levels'
	elif name == 'temperature':
		return 'air_temperature'
	elif name == 'change_over_time_in_air_potential_temperature':
		return 'tendency_of_air_potential_temperature'
	elif name == 'exner_function':
		return 'exner_function_on_interface_levels'
	elif name == 'height':
		return 'height_on_interface_levels'
	elif name == 'x_velocity':
		return 'x_velocity_at_u_locations'
	elif name == 'y_velocity':
		return 'y_velocity_at_v_locations'
	else:
		return name


def convert_old_to_new_pickle(filename, filename_new=None, apply_name_conversion=True):
	root, ext = os.path.splitext(filename)
	if ext != '.pickle':
		return

	if filename_new is None:
		filename_new = root + '_new.pickle'

	with open(filename, 'rb') as data:
		states_old = pickle.load(data)
		state_names = states_old.variable_names
		nt = states_old[state_names[0]].shape[3]

		try:
			diagnostics_old = pickle.load(data)
			diagnostics_names = diagnostics_old.variable_names
		except EOFError:
			diagnostics_names = []

		states_new = []
		for t in range(nt):
			current_time = states_old[state_names[0]].coords['time'][t]
			state = {'time': cd64td(current_time)}
			for state_name in state_names:
				state_name_new = convert_variable_name(state_name) if apply_name_conversion \
					else state_name
				state[state_name_new] = states_old[state_name][:, :, :, t]
			for diagnostic_name in diagnostics_names:
				diagnostic_name_new = convert_variable_name(diagnostic_name) if apply_name_conversion \
					else diagnostic_name
				try:
					state[diagnostic_name_new] = \
						diagnostics_old[diagnostic_name].loc[:, :, :, current_time]
				except IndexError:
					state[diagnostic_name_new] = None
				except KeyError:
					state[diagnostic_name_new] = None
			states_new.append(copy.deepcopy(state))

			if t == nt-1:
				grid = copy.deepcopy(states_old.grid)

	with open(filename_new, 'wb') as output:
		pickle.dump(grid, output)
		pickle.dump(states_new, output)

	return filename_new


def convert_grid(grid):
	domain_x = DataArray([grid.x.values[0], grid.x.values[-1]], dims='x',
						 attrs={'units': grid.x.attrs['units']})
	nx = grid.nx
	domain_y = DataArray([grid.y.values[0], grid.y.values[-1]], dims='y',
						 attrs={'units': grid.y.attrs['units']})
	ny = grid.ny
	domain_z = DataArray([grid.z_half_levels.values[0], grid.z_half_levels.values[-1]],
						 dims='air_potential_temperature',
						 attrs={'units': grid.z.attrs['units']})
	nz = grid.nz

	try:
		topo = grid.topography
	except AttributeError:
		topo = grid._topography

	try:
		topo_kwargs_ = topo.topo_kwargs
		topo_kwargs  = {
			'topo_max_height': DataArray(topo_kwargs_['topo_max_height'],
										 attrs={'units': 'm'}),
			'topo_width_x': DataArray(topo_kwargs_['topo_width_x'],
									  attrs={'units': 'm'}),
			'topo_width_y': DataArray(topo_kwargs_['topo_width_y'],
									  attrs={'units': 'm'}),
			'topo_center_x': DataArray(topo_kwargs_['topo_center_x'],
									   attrs={'units': 'm'}),
			'topo_center_y': DataArray(topo_kwargs_['topo_center_y'],
									   attrs={'units': 'm'}),
		}
	except AttributeError:
		try:
			topo_kwargs_ = topo._topo_kwargs
			topo_kwargs  = {
				'topo_max_height': DataArray(topo_kwargs_['topo_max_height'],
											 attrs={'units': 'm'}),
				'topo_width_x': DataArray(topo_kwargs_['topo_width_x'],
										  attrs={'units': 'm'}),
				'topo_width_y': DataArray(topo_kwargs_['topo_width_y'],
										  attrs={'units': 'm'}),
				'topo_center_x': DataArray(topo_kwargs_['topo_center_x'],
										   attrs={'units': 'm'}),
				'topo_center_y': DataArray(topo_kwargs_['topo_center_y'],
										   attrs={'units': 'm'}),
			}
		except AttributeError:
			print('>>>>>>>>>> HERE AGAIN <<<<<<<<<<')
			topo_kwargs = {}

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type=topo.topo_type, topo_time=topo.topo_time,
				topo_kwargs=topo_kwargs, dtype=domain_x.values.dtype)


def get_shape(key, grid):
	if key in ['accumulated_precipitation', 'air_density', 'air_isentropic_density',
			   'air_temperature', 'mass_fraction_of_water_vapor_in_air',
			   'mass_fraction_of_cloud_liquid_water_in_air',
			   'mass_fraction_of_precipitation_water_in_air', 'montgomery_potential',
			   'precipitation', 'tendency_of_air_potential_temperature',
			   'x_momentum_isentropic', 'y_momentum_isentropic']:
		return (grid.nx, grid.ny, grid.nz)
	elif key in ['air_pressure_on_interface_levels', 'exner_function_on_interface_levels',
				 'height_on_interface_levels']:
		return (grid.nx, grid.ny, grid.nz+1)
	elif key in ['x_velocity_at_u_locations']:
		return (grid.nx+1, grid.ny, grid.nz)
	elif key in ['y_velocity_at_v_locations']:
		return (grid.nx, grid.ny+1, grid.nz)
	else:
		raise KeyError('Unknown key {}.'.format(key))


def get_default_units(key):
	if key == 'accumulated_precipitation':
		return 'mm'
	elif key == 'air_density':
		return 'kg m^-3'
	elif key == 'air_isentropic_density':
		return 'kg m^-2 K^-1'
	elif key == 'air_pressure_on_interface_levels':
		return 'Pa'
	elif key == 'air_temperature':
		return 'K'
	elif key == 'exner_function_on_interface_levels':
		return 'm^2 s^-2 K^-1'
	elif key == 'height_on_interface_levels':
		return 'm'
	elif key == 'mass_fraction_of_water_vapor_in_air':
		return 'g g^-1'
	elif key == 'mass_fraction_of_cloud_liquid_water_in_air':
		return 'g g^-1'
	elif key == 'mass_fraction_of_precipitation_water_in_air':
		return 'g g^-1'
	elif key == 'montgomery_potential':
		return 'm^2 s^-2'
	elif key == 'precipitation':
		return 'mm h^-1'
	elif key == 'tendency_of_air_potential_temperature':
		return 'K s^-1'
	elif key == 'x_momentum_isentropic':
		return 'kg m^-1 K^-1 s^-1'
	elif key == 'x_velocity_at_u_locations':
		return 'm s^-1'
	elif key == 'y_momentum_isentropic':
		return 'kg m^-1 K^-1 s^-1'
	elif key == 'y_velocity_at_v_locations':
		return 'm s^-1'
	else:
		raise KeyError('Unknown key {}.'.format(key))


def convert_pickle_to_netcdf(filename, filename_new=None):
	root, ext = os.path.splitext(filename)
	if ext != '.pickle':
		return

	if filename_new is None:
		filename_new = root + '_new.nc'
	root, ext = os.path.splitext(filename_new)
	if ext != '.nc':
		return

	with open(filename, 'rb') as data:
		grid_  = pickle.load(data)
		grid   = convert_grid(grid_)

		monitor = NetCDFMonitor(filename_new, grid)

		states = pickle.load(data)
		for state_ in states:
			state = {'time': state_['time']}

			for key_ in state_.keys():
				if key_ != 'time':
					key 	  = convert_variable_name(key_)
					raw_array = np.zeros(get_shape(key, grid)) if state_[key_] is None \
						else state_[key_].values
					units     = get_default_units(key) if state_[key_] is None \
						else state_[key_].attrs['units']

					state[key] = make_data_array_3d(raw_array, grid, units, name=key)

			monitor.store(state)

		monitor.write()
