"""
This module contains:
	get_isothermal_isentropic_state
"""
import numpy as np
from sympl import DataArray

from tasmania.utils.data_utils import get_physical_constants, make_data_array_3d

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


_d_physical_constants = {
    'gas_constant_of_dry_air':
    	DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
	'gravitational_acceleration':
		DataArray(9.81, attrs={'units': 'm s^-2'}),
	'reference_air_pressure':
		DataArray(1.0e5, attrs={'units': 'Pa'}),
    'specific_heat_of_dry_air_at_constant_pressure':
    	DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
}


def get_isothermal_isentropic_state(grid, time, x_velocity, y_velocity, temperature,
									dtype=datatype, physical_constants=None):
	# Shortcuts
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.values.item()

	# Get needed physical constants
	pcs  = get_physical_constants(_d_physical_constants, physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# Initialize the velocity components
	u = x_velocity.to_units('m s^-1').values.item() * np.ones((nx+1, ny, nz), dtype=dtype)
	v = y_velocity.to_units('m s^-1').values.item() * np.ones((nx, ny+1, nz), dtype=dtype)

	# Initialize the air pressure
	theta1d = grid.z_on_interface_levels.values[np.newaxis, np.newaxis, :]
	theta   = np.tile(theta1d, (nx, ny, 1))
	temp	= temperature.to_units('K').values.item()
	p       = pref * ((temp / theta) ** (cp / Rd))

	# Initialize the Exner function
	exn = cp * temp / theta

	# Diagnose the Montgomery potential
	hs = grid.topography_height
	mtg_s = cp * temp + g * hs
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]

	# Diagnose the height of the half levels
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = hs
	for k in range(nz-2, -1, -1):
		h[:, :, k] = h[:, :, k+1] - Rd / (cp * g) * \
					 (theta[:, :, k] * exn[:, :, k] + theta[:, :, k+1] * exn[:, :, k+1]) * \
					 (p[:, :, k] - p[:, :, k+1]) / (p[:, :, k] - p[:, :, k+1])

	# Diagnose the isentropic density amd the momentums
	s  = - (p[:, :, :-1] - p[:, :, 1:]) / (g * dz)
	su = 0.5 * s * (u[:-1, :, :] + u[1:, :, :])
	sv = 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])

	# Instantiate the return state
	state = {
		'time': time,
		'air_isentropic_density':
			make_data_array_3d(s, grid, 'kg m^-2 K^-1', name='air_isentropic_density'),
		'air_pressure_on_interface_levels':
			make_data_array_3d(p, grid, 'Pa', name='air_pressure_on_interface_levels'),
		'exner_function_on_interface_levels':
			make_data_array_3d(exn, grid, 'J K^-1 kg^-1', name='exner_function_on_interface_levels'),
		'height_on_interface_levels':
			make_data_array_3d(h, grid, 'm', name='height_on_interface_levels'),
		'montgomery_potential':
			make_data_array_3d(mtg, grid, 'J kg^-1', name='montgomery_potential'),
		'x_momentum_isentropic':
			make_data_array_3d(su, grid, 'kg m^-1 K^-1 s^-1', name='x_momentum_isentropic'),
		'x_velocity_at_u_locations':
			make_data_array_3d(u, grid, 'm s^-1', name='x_velocity_at_u_locations'),
		'y_momentum_isentropic':
			make_data_array_3d(sv, grid, 'kg m^-1 K^-1 s^-1', name='y_momentum_isentropic'),
		'y_velocity_at_v_locations':
			make_data_array_3d(v, grid, 'm s^-1', name='y_velocity_at_v_locations'),
	}

	return state
