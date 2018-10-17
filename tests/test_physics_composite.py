import copy
from datetime import timedelta
import numpy as np
import pytest

import gridtools as gt
from tasmania.core.physics_composite import DiagnosticComponentComposite, \
											ConcurrentCoupling, \
											ParallelSplitting, \
											SequentialUpdateSplitting
from tasmania.physics.isentropic_diagnostics import IsentropicDiagnostics, \
													IsentropicVelocityComponents
from tasmania.physics.isentropic_tendencies import ConservativeIsentropicPressureGradient, \
												   NonconservativeIsentropicPressureGradient
from tasmania.physics.microphysics import Kessler, SaturationAdjustmentKessler


def test_diagnostic(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float64

	dv = IsentropicDiagnostics(grid, True,
							   pt=state['air_pressure_on_interface_levels'][0, 0, 0],
							   backend=backend, dtype=dtype)
	sa = SaturationAdjustmentKessler(grid, backend=backend)

	dcc = DiagnosticComponentComposite(dv, sa)

	assert 'air_isentropic_density' in dcc.input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.input_properties
	assert len(dcc.input_properties) == 3

	assert 'air_density' in dcc.diagnostic_properties
	assert 'air_pressure_on_interface_levels' in dcc.diagnostic_properties
	assert 'air_temperature' in dcc.diagnostic_properties
	assert 'exner_function_on_interface_levels' in dcc.diagnostic_properties
	assert 'height_on_interface_levels' in dcc.diagnostic_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.diagnostic_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.diagnostic_properties
	assert 'montgomery_potential' in dcc.diagnostic_properties
	assert len(dcc.diagnostic_properties) == 8

	assert 'air_density' in dcc.output_properties
	assert 'air_isentropic_density' in dcc.output_properties
	assert 'air_pressure_on_interface_levels' in dcc.output_properties
	assert 'air_temperature' in dcc.output_properties
	assert 'exner_function_on_interface_levels' in dcc.output_properties
	assert 'height_on_interface_levels' in dcc.output_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.output_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.output_properties
	assert 'montgomery_potential' in dcc.output_properties
	assert len(dcc.output_properties) == 9

	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')

	diagnostics = dcc(state)

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   diagnostics['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   diagnostics['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   diagnostics['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   diagnostics['montgomery_potential'])


def test_cc_serial(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, False,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed', backend, dtype)

	cc = ConcurrentCoupling(diags, pg, mode='serial')
	assert isinstance(cc, ConcurrentCoupling)

	assert cc.provisional_state_input_properties == {}
	assert cc.provisional_state_output_properties == {}

	assert 'air_isentropic_density' in cc.current_state_input_properties
	assert len(cc.current_state_input_properties) == 1

	assert 'air_isentropic_density' in cc.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in cc.current_state_output_properties
	assert 'exner_function_on_interface_levels' in cc.current_state_output_properties
	assert 'height_on_interface_levels' in cc.current_state_output_properties
	assert 'montgomery_potential' in cc.current_state_output_properties
	assert len(cc.current_state_output_properties) == 5

	assert 'x_momentum_isentropic' in cc.tendency_properties
	assert 'y_momentum_isentropic' in cc.tendency_properties
	assert len(cc.tendency_properties) == 2

	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	tendencies = cc(state=state)

	assert 'time' in tendencies
	assert 'x_momentum_isentropic' in tendencies
	assert 'y_momentum_isentropic' in tendencies
	assert len(tendencies) == 3

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])


def test_cc_serial_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float64

	diags = IsentropicDiagnostics(grid, True,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg = NonconservativeIsentropicPressureGradient(grid, 4, 'relaxed', backend, dtype)
	kessler = Kessler(grid, backend=backend)
	sa = SaturationAdjustmentKessler(grid, backend=backend)

	cc = ConcurrentCoupling(diags, pg, kessler, sa, mode='serial')
	assert isinstance(cc, ConcurrentCoupling)

	assert cc.provisional_state_input_properties == {}
	assert cc.provisional_state_output_properties == {}

	assert 'air_isentropic_density' in cc.current_state_input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in cc.current_state_input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in cc.current_state_input_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in cc.current_state_input_properties
	assert len(cc.current_state_input_properties) == 4

	assert 'air_density' in cc.current_state_output_properties
	assert 'air_isentropic_density' in cc.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in cc.current_state_output_properties
	assert 'air_temperature' in cc.current_state_output_properties
	assert 'exner_function_on_interface_levels' in cc.current_state_output_properties
	assert 'height_on_interface_levels' in cc.current_state_output_properties
	assert 'mass_fraction_of_water_vapor_in_air' in cc.current_state_input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in cc.current_state_input_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in cc.tendency_properties
	assert 'montgomery_potential' in cc.current_state_output_properties
	assert len(cc.current_state_output_properties) == 10

	assert 'air_potential_temperature' in cc.tendency_properties
	assert 'mass_fraction_of_water_vapor_in_air' in cc.tendency_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in cc.tendency_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in cc.tendency_properties
	assert 'x_velocity' in cc.tendency_properties
	assert 'y_velocity' in cc.tendency_properties
	assert len(cc.tendency_properties) == 6

	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	tendencies = cc(state=state)

	assert 'time' in tendencies
	assert 'air_potential_temperature' in tendencies
	assert 'mass_fraction_of_water_vapor_in_air' in tendencies
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in tendencies
	assert 'mass_fraction_of_precipitation_water_in_air' in tendencies
	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 7

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])


def test_cc_asparallel(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, False,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = NonconservativeIsentropicPressureGradient(grid, 4, 'relaxed', backend, dtype)

	cc = ConcurrentCoupling(diags, pg, mode='asparallel')
	assert isinstance(cc, ConcurrentCoupling)

	assert cc.provisional_state_input_properties == {}
	assert cc.provisional_state_output_properties == {}

	assert 'air_isentropic_density' in cc.current_state_input_properties
	assert 'montgomery_potential' in cc.current_state_input_properties
	assert len(cc.current_state_input_properties) == 2

	assert 'air_isentropic_density' in cc.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in cc.current_state_output_properties
	assert 'exner_function_on_interface_levels' in cc.current_state_output_properties
	assert 'height_on_interface_levels' in cc.current_state_output_properties
	assert 'montgomery_potential' in cc.current_state_output_properties
	assert len(cc.current_state_output_properties) == 5

	assert 'x_velocity' in cc.tendency_properties
	assert 'y_velocity' in cc.tendency_properties
	assert len(cc.tendency_properties) == 2

	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	tendencies = cc(state=state)

	assert 'time' in tendencies
	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 3

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])


def test_ps_serial(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = copy.deepcopy(states[-1])
	state_prv = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, False,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed', backend, dtype)

	dt = timedelta(seconds=10)

	#
	# retrieve_diagnostics_from_provisional_state=False
	#
	ps = ParallelSplitting(diags, pg, mode='serial', time_integration_scheme='rk3',
						   grid=grid, retrieve_diagnostics_from_provisional_state=False)
	assert isinstance(ps, ParallelSplitting)

	assert 'air_isentropic_density' in ps.current_state_input_properties
	assert 'x_momentum_isentropic' in ps.current_state_input_properties
	assert 'y_momentum_isentropic' in ps.current_state_input_properties
	assert len(ps.current_state_input_properties) == 3

	assert 'x_momentum_isentropic' in ps.current_state_input_properties
	assert 'y_momentum_isentropic' in ps.current_state_input_properties
	assert len(ps.provisional_state_input_properties) == 2

	assert 'air_isentropic_density' in ps.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in ps.current_state_output_properties
	assert 'exner_function_on_interface_levels' in ps.current_state_output_properties
	assert 'height_on_interface_levels' in ps.current_state_output_properties
	assert 'montgomery_potential' in ps.current_state_output_properties
	assert 'x_momentum_isentropic' in ps.current_state_output_properties
	assert 'y_momentum_isentropic' in ps.current_state_output_properties
	assert len(ps.current_state_output_properties) == 7

	assert 'x_momentum_isentropic' in ps.current_state_output_properties
	assert 'y_momentum_isentropic' in ps.current_state_output_properties
	assert len(ps.provisional_state_output_properties) == 2

	assert ps.tendency_properties == {}

	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')

	tendencies = ps(state=state, state_prv=state_prv, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])

	#
	# retrieve_diagnostics_from_provisional_state=True
	#
	ps = ParallelSplitting(diags, pg, mode='serial', time_integration_scheme='rk3',
						   grid=grid, retrieve_diagnostics_from_provisional_state=True)
	assert isinstance(ps, ParallelSplitting)

	assert 'air_isentropic_density' in ps.current_state_input_properties
	assert 'montgomery_potential' in ps.current_state_input_properties
	assert 'x_momentum_isentropic' in ps.current_state_input_properties
	assert 'y_momentum_isentropic' in ps.current_state_input_properties
	assert len(ps.current_state_input_properties) == 4

	assert 'air_isentropic_density' in ps.provisional_state_input_properties
	assert 'x_momentum_isentropic' in ps.provisional_state_input_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_input_properties
	assert len(ps.provisional_state_input_properties) == 3

	assert 'air_isentropic_density' in ps.current_state_output_properties
	assert 'montgomery_potential' in ps.current_state_output_properties
	assert 'x_momentum_isentropic' in ps.current_state_output_properties
	assert 'y_momentum_isentropic' in ps.current_state_output_properties
	assert len(ps.current_state_output_properties) == 4

	assert 'air_isentropic_density' in ps.provisional_state_output_properties
	assert 'air_pressure_on_interface_levels' in ps.provisional_state_output_properties
	assert 'exner_function_on_interface_levels' in ps.provisional_state_output_properties
	assert 'height_on_interface_levels' in ps.provisional_state_output_properties
	assert 'montgomery_potential' in ps.provisional_state_output_properties
	assert 'x_momentum_isentropic' in ps.provisional_state_output_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_output_properties
	assert len(ps.provisional_state_output_properties) == 7

	assert ps.tendency_properties == {}

	state_prv.pop('air_pressure_on_interface_levels')
	state_prv.pop('exner_function_on_interface_levels')
	state_prv.pop('height_on_interface_levels')
	state_prv.pop('montgomery_potential')

	tendencies = ps(state=state, state_prv=state_prv, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state_prv['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state_prv['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state_prv['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state_prv['montgomery_potential'],
					   states[-1]['montgomery_potential'])


def test_ps_asparallel(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = copy.deepcopy(states[-1])
	state_prv = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, False,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed', backend, dtype)

	dt = timedelta(seconds=10)

	ps = ParallelSplitting(diags, pg, mode='asparallel', time_integration_scheme='rk3',
						   grid=grid, retrieve_diagnostics_from_provisional_state=False)
	assert isinstance(ps, ParallelSplitting)

	assert 'air_isentropic_density' in ps.current_state_input_properties
	assert 'montgomery_potential' in ps.current_state_input_properties
	assert 'x_momentum_isentropic' in ps.current_state_input_properties
	assert 'y_momentum_isentropic' in ps.current_state_input_properties
	assert len(ps.current_state_input_properties) == 4

	assert 'x_momentum_isentropic' in ps.provisional_state_input_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_input_properties
	assert len(ps.provisional_state_input_properties) == 2

	assert 'air_isentropic_density' in ps.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in ps.current_state_output_properties
	assert 'exner_function_on_interface_levels' in ps.current_state_output_properties
	assert 'height_on_interface_levels' in ps.current_state_output_properties
	assert 'montgomery_potential' in ps.current_state_output_properties
	assert 'x_momentum_isentropic' in ps.current_state_output_properties
	assert 'y_momentum_isentropic' in ps.current_state_output_properties
	assert len(ps.current_state_output_properties) == 7

	assert 'x_momentum_isentropic' in ps.provisional_state_output_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_output_properties
	assert len(ps.provisional_state_output_properties) == 2

	assert ps.tendency_properties == {}

	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')

	tendencies = ps(state=state, state_prv=state_prv, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])


def test_ps_asparallel_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = copy.deepcopy(states[-1])
	state_prv = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float64

	diags = IsentropicDiagnostics(grid, True,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg = ConservativeIsentropicPressureGradient(grid, 4, 'relaxed', backend, dtype)
	kessler = Kessler(grid, tendency_of_air_potential_temperature_in_diagnostics=True,
					  backend=backend)
	sa = SaturationAdjustmentKessler(grid, backend=backend)

	dt = timedelta(seconds=10)

	ps = ParallelSplitting(diags, pg, kessler, sa, mode='as_parallel',
						   grid=grid, time_integration_scheme='rk2')
	assert isinstance(ps, ParallelSplitting)

	assert 'air_density' in ps.current_state_input_properties
	assert 'air_isentropic_density' in ps.current_state_input_properties
	assert 'air_pressure_on_interface_levels' in ps.current_state_input_properties
	assert 'air_temperature' in ps.current_state_input_properties
	assert 'exner_function_on_interface_levels' in ps.current_state_input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in ps.current_state_input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in ps.current_state_input_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in ps.current_state_input_properties
	assert 'montgomery_potential' in ps.current_state_input_properties
	assert 'x_momentum_isentropic' in ps.current_state_input_properties
	assert 'y_momentum_isentropic' in ps.current_state_input_properties
	assert len(ps.current_state_input_properties) == 11

	assert 'mass_fraction_of_water_vapor_in_air' in ps.provisional_state_input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in ps.provisional_state_input_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in ps.provisional_state_input_properties
	assert 'x_momentum_isentropic' in ps.provisional_state_input_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_input_properties
	assert len(ps.provisional_state_input_properties) == 5

	assert 'air_density' in ps.current_state_output_properties
	assert 'air_isentropic_density' in ps.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in ps.current_state_output_properties
	assert 'air_temperature' in ps.current_state_output_properties
	assert 'exner_function_on_interface_levels' in ps.current_state_output_properties
	assert 'height_on_interface_levels' in ps.current_state_output_properties
	assert 'mass_fraction_of_water_vapor_in_air' in ps.current_state_output_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in ps.current_state_output_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in ps.current_state_output_properties
	assert 'montgomery_potential' in ps.current_state_output_properties
	assert 'tendency_of_air_potential_temperature' in ps.current_state_output_properties
	assert 'x_momentum_isentropic' in ps.current_state_output_properties
	assert 'y_momentum_isentropic' in ps.current_state_output_properties
	assert len(ps.current_state_output_properties) == 13

	assert 'mass_fraction_of_water_vapor_in_air' in ps.provisional_state_output_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in ps.provisional_state_output_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in ps.provisional_state_output_properties
	assert 'x_momentum_isentropic' in ps.provisional_state_output_properties
	assert 'y_momentum_isentropic' in ps.provisional_state_output_properties
	assert len(ps.provisional_state_output_properties) == 5

	assert ps.tendency_properties == {}

	tendencies = ps(state=state, state_prv=state_prv, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])


def test_sus(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, False,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed', backend, dtype)
	vc	  = IsentropicVelocityComponents(grid, 'relaxed', state,
										 backend=backend, dtype=dtype)

	dt = timedelta(seconds=10)

	sus = SequentialUpdateSplitting(diags, pg, vc, time_integration_scheme='rk3', grid=grid)
	assert isinstance(sus, SequentialUpdateSplitting)

	assert 'air_isentropic_density' in sus.current_state_input_properties
	assert 'x_momentum_isentropic' in sus.current_state_input_properties
	assert 'y_momentum_isentropic' in sus.current_state_input_properties
	assert len(sus.current_state_input_properties) == 3

	assert sus.provisional_state_input_properties == sus.current_state_input_properties

	assert 'air_isentropic_density' in sus.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in sus.current_state_output_properties
	assert 'exner_function_on_interface_levels' in sus.current_state_output_properties
	assert 'height_on_interface_levels' in sus.current_state_output_properties
	assert 'montgomery_potential' in sus.current_state_output_properties
	assert 'x_momentum_isentropic' in sus.current_state_output_properties
	assert 'x_velocity_at_u_locations' in sus.current_state_output_properties
	assert 'y_momentum_isentropic' in sus.current_state_output_properties
	assert 'y_velocity_at_v_locations' in sus.current_state_output_properties
	assert len(sus.current_state_output_properties) == 9

	assert sus.provisional_state_output_properties == sus.current_state_output_properties

	assert sus.tendency_properties == {}

	#
	# state
	#
	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')
	state.pop('x_velocity_at_u_locations')
	state.pop('y_velocity_at_v_locations')

	tendencies = sus(state=state, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])

	#
	# state_prv
	#
	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')
	state.pop('x_velocity_at_u_locations')
	state.pop('y_velocity_at_v_locations')

	tendencies = sus(state_prv=state, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])


def test_sus_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = copy.deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diags = IsentropicDiagnostics(grid, True,
								  pt=state['air_pressure_on_interface_levels'][0, 0, 0],
								  backend=backend, dtype=dtype)
	pg	  = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed', backend, dtype)
	vc	  = IsentropicVelocityComponents(grid, 'relaxed', state,
										 backend=backend, dtype=dtype)
	ks	  = Kessler(grid, tendency_of_air_potential_temperature_in_diagnostics=True,
					backend=backend)
	sa 	  = SaturationAdjustmentKessler(grid, backend=backend)

	dt = timedelta(seconds=10)

	sus = SequentialUpdateSplitting(diags, pg, vc, ks, sa,
									time_integration_scheme='rk3', grid=grid)
	assert isinstance(sus, SequentialUpdateSplitting)

	assert 'air_isentropic_density' in sus.current_state_input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in sus.current_state_input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in sus.current_state_input_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in sus.current_state_input_properties
	assert 'x_momentum_isentropic' in sus.current_state_input_properties
	assert 'y_momentum_isentropic' in sus.current_state_input_properties
	assert len(sus.current_state_input_properties) == 6

	assert sus.provisional_state_input_properties == sus.current_state_input_properties

	assert 'air_density' in sus.current_state_output_properties
	assert 'air_isentropic_density' in sus.current_state_output_properties
	assert 'air_pressure_on_interface_levels' in sus.current_state_output_properties
	assert 'air_temperature' in sus.current_state_output_properties
	assert 'exner_function_on_interface_levels' in sus.current_state_output_properties
	assert 'height_on_interface_levels' in sus.current_state_output_properties
	assert 'mass_fraction_of_water_vapor_in_air' in sus.current_state_output_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in sus.current_state_output_properties
	assert 'mass_fraction_of_precipitation_water_in_air' in sus.current_state_output_properties
	assert 'montgomery_potential' in sus.current_state_output_properties
	assert 'tendency_of_air_potential_temperature' in sus.current_state_output_properties
	assert 'x_momentum_isentropic' in sus.current_state_output_properties
	assert 'x_velocity_at_u_locations' in sus.current_state_output_properties
	assert 'y_momentum_isentropic' in sus.current_state_output_properties
	assert 'y_velocity_at_v_locations' in sus.current_state_output_properties
	assert len(sus.current_state_output_properties) == 15

	assert sus.provisional_state_output_properties == sus.current_state_output_properties

	assert sus.tendency_properties == {}

	#
	# state
	#
	state.pop('air_density')
	state.pop('air_pressure_on_interface_levels')
	state.pop('air_temperature')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')
	state.pop('x_velocity_at_u_locations')
	state.pop('y_velocity_at_v_locations')

	tendencies = sus(state=state, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_density'],
					   states[-1]['air_density'])
	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['air_temperature'],
					   states[-1]['air_temperature'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])

	#
	# state_prv
	#
	state.pop('air_density')
	state.pop('air_pressure_on_interface_levels')
	state.pop('air_temperature')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')
	state.pop('x_velocity_at_u_locations')
	state.pop('y_velocity_at_v_locations')

	tendencies = sus(state_prv=state, timestep=dt)

	assert 'time' in tendencies
	assert len(tendencies) == 1

	assert np.allclose(state['air_density'],
					   states[-1]['air_density'])
	assert np.allclose(state['air_pressure_on_interface_levels'],
					   states[-1]['air_pressure_on_interface_levels'])
	assert np.allclose(state['air_temperature'],
					   states[-1]['air_temperature'])
	assert np.allclose(state['exner_function_on_interface_levels'],
					   states[-1]['exner_function_on_interface_levels'])
	assert np.allclose(state['height_on_interface_levels'],
					   states[-1]['height_on_interface_levels'])
	assert np.allclose(state['montgomery_potential'],
					   states[-1]['montgomery_potential'])


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data, isentropic_moist_data
	#test_parallel_splitting_asparallel_moist(isentropic_moist_data())
