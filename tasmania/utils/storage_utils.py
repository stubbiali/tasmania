import copy
import os
import pickle

from tasmania.utils.utils import convert_datetime64_to_datetime as cd64td


def convert_variable_name(name):
	if name == 'isentropic_density':
		return 'air_isentropic_density'
	elif name == 'pressure' or name == 'air_pressure':
		return 'air_pressure_on_interface_levels'
	elif name == 'temperature':
		return 'air_temperature'
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


def convert_dataset(filename, filename_new = None, apply_name_conversion = True):
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
				state_name_new = convert_variable_name(state_name) if apply_name_conversion else state_name
				state[state_name_new] = states_old[state_name][:,:,:,t]
			for diagnostic_name in diagnostics_names:
				diagnostic_name_new = convert_variable_name(diagnostic_name) if apply_name_conversion else diagnostic_name
				try:
					state[diagnostic_name_new] = diagnostics_old[diagnostic_name].loc[:,:,:,current_time]
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


if __name__ == '__main__':
	path_old = os.path.join(os.environ['TASMANIA_ROOT'], 'data/old_datasets')
	filenames_ = [f for f in os.listdir(path_old) if os.path.isfile(os.path.join(path_old, f))]
	filenames_old = [os.path.join(path_old, f) for f in filenames_]

	path_new = os.path.join(os.environ['TASMANIA_ROOT'], 'data')
	filenames_new = [os.path.join(path_new, f) for f in filenames_]

	for filename_old, filename_new in zip(filenames_old, filenames_new):
		convert_dataset(filename_old, filename_new, apply_name_conversion = True)
		print('File {} processed.'.format(filename_new))

