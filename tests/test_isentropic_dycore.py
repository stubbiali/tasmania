from datetime import timedelta
import numpy as np
import pytest
from sympl import DataArray

import gridtools as gt
from tasmania.dynamics.isentropic_dycore import IsentropicDynamicalCore
from tasmania.physics.microphysics import Kessler, SaturationAdjustmentKessler, \
										  RaindropFallVelocity


def test_dry(isentropic_dry_data):
	grid  	    = isentropic_dry_data[0]
	state       = isentropic_dry_data[1][0]
	state_final = isentropic_dry_data[1][-1]

	dycore = IsentropicDynamicalCore(grid, moist_on=False,
									 time_integration_scheme='forward_euler',
									 horizontal_flux_scheme='maccormack',
									 horizontal_boundary_type='relaxed',
									 damp_on=True, damp_type='rayleigh', damp_depth=15,
									 damp_max=0.0002, damp_at_every_stage=False,
									 smooth_on=True, smooth_type='first_order',
									 smooth_coeff=0.12, smooth_at_every_stage=False,
									 backend=gt.mode.NUMPY, dtype=np.float32)

	timestep   = timedelta(seconds=24)
	niter      = 1800

	for i in range(niter):
		dycore.update_topography((i+1)*timestep)

		state_new = dycore(state, {}, timestep)
		state.update(state_new)

	for key in state.keys():
		if key != 'time':
			assert np.allclose(state_final[key].values, state[key].values)


def test_moist(isentropic_moist_data):
	grid  	    = isentropic_moist_data[0]
	state       = isentropic_moist_data[1][0]
	state_final = isentropic_moist_data[1][-1]

	kessler = Kessler(grid, air_pressure_on_interface_levels=True,
					  rain_evaporation=False, backend=gt.mode.NUMPY)

	saturation = SaturationAdjustmentKessler(grid, air_pressure_on_interface_levels=True,
											 backend=gt.mode.NUMPY)

	dycore = IsentropicDynamicalCore(grid, moist_on=True,
									 time_integration_scheme='forward_euler',
									 horizontal_flux_scheme='upwind',
									 horizontal_boundary_type='relaxed',
									 damp_on=False,
									 smooth_on=True, smooth_type='first_order',
									 smooth_coeff=0.20, smooth_coeff_max=1.00,
									 smooth_damp_depth=30, smooth_at_every_stage=True,
									 adiabatic_flow=True,
									 backend=gt.mode.NUMPY, dtype=np.float64)

	timestep   = timedelta(seconds=10)
	niter      = 2160

	for i in range(niter):
		dycore.update_topography((i+1)*timestep)

		tendencies, _ = kessler(state)

		state_new = dycore(state, tendencies, timestep)
		state.update(state_new)

		state_saturated = saturation(state)
		state.update(state_saturated)

	for key in state.keys():
		if key != 'time':
			assert np.allclose(state_final[key].values, state[key].values)


def test_moist_sedimentation(isentropic_moist_sedimentation_data):
	grid  	    = isentropic_moist_sedimentation_data[0]
	state       = isentropic_moist_sedimentation_data[1][0]
	state_final = isentropic_moist_sedimentation_data[1][-1]

	kessler = Kessler(grid, air_pressure_on_interface_levels=True,
					  rain_evaporation=False, backend=gt.mode.NUMPY,
					  autoconversion_threshold=DataArray(0.1, attrs={'units': 'g kg^-1'}))

	rfv = RaindropFallVelocity(grid, backend=gt.mode.NUMPY)

	saturation = SaturationAdjustmentKessler(grid, air_pressure_on_interface_levels=True,
											 backend=gt.mode.NUMPY)

	dycore = IsentropicDynamicalCore(grid, moist_on=True,
									 time_integration_scheme='forward_euler',
									 horizontal_flux_scheme='maccormack',
									 horizontal_boundary_type='relaxed',
									 smooth_on=True, smooth_type='first_order',
									 smooth_coeff=0.20, smooth_at_every_stage=True,
									 adiabatic_flow=True, sedimentation_on=True,
									 sedimentation_flux_type='second_order_upwind',
									 sedimentation_substeps=2,
									 raindrop_fall_velocity_diagnostic=rfv,
									 backend=gt.mode.NUMPY, dtype=np.float64)

	timestep   = timedelta(seconds=10)
	niter      = 2160

	for i in range(niter):
		dycore.update_topography((i+1)*timestep)

		tendencies, _ = kessler(state)

		state_new = dycore(state, tendencies, timestep)
		state.update(state_new)

		state_saturated = saturation(state)
		state.update(state_saturated)

	for key in state.keys():
		if key != 'time':
			assert np.allclose(state_final[key].values, state[key].values)


def test_moist_sedimentation_evaporation(isentropic_moist_sedimentation_evaporation_data):
	grid  	    = isentropic_moist_sedimentation_evaporation_data[0]
	state       = isentropic_moist_sedimentation_evaporation_data[1][0]
	state_final = isentropic_moist_sedimentation_evaporation_data[1][-1]

	kessler = Kessler(grid, air_pressure_on_interface_levels=True,
					  rain_evaporation=True, backend=gt.mode.NUMPY,
					  autoconversion_threshold=DataArray(0.1, attrs={'units': 'g kg^-1'}))

	rfv = RaindropFallVelocity(grid, backend=gt.mode.NUMPY)

	saturation = SaturationAdjustmentKessler(grid, air_pressure_on_interface_levels=True,
											 backend=gt.mode.NUMPY)

	dycore = IsentropicDynamicalCore(grid, moist_on=True,
									 time_integration_scheme='centered',
									 horizontal_flux_scheme='centered',
									 horizontal_boundary_type='relaxed',
									 smooth_on=True, smooth_type='first_order',
									 smooth_coeff=0.20, smooth_at_every_stage=True,
									 adiabatic_flow=True, sedimentation_on=True,
									 sedimentation_flux_type='second_order_upwind',
									 sedimentation_substeps=2,
									 raindrop_fall_velocity_diagnostic=rfv,
									 backend=gt.mode.NUMPY, dtype=np.float64)

	timestep   = timedelta(seconds=10)
	niter      = 2160

	for i in range(niter):
		dycore.update_topography((i+1)*timestep)

		tendencies, _ = kessler(state)

		state_new = dycore(state, tendencies, timestep)
		state.update(state_new)

		state_saturated = saturation(state)
		state.update(state_saturated)

	for key in state.keys():
		if key != 'time':
			assert np.allclose(state_final[key].values, state[key].values)


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_moist_data
	#test_moist(isentropic_moist_data())
