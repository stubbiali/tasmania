from datetime import timedelta
import gridtools as gt
import numpy as np
import tasmania as taz


grid, states = taz.load_netcdf_dataset('../tests/baseline_datasets/isentropic_dry.nc')
state = states[0]

pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt, 
							   backend=gt.mode.NUMPY, dtype=np.float32)

pg = taz.NonconservativeIsentropicPressureGradient(grid, order=2, horizontal_boundary_type='relaxed',
							   					   backend=gt.mode.NUMPY, dtype=np.float32)

dt = timedelta(seconds=24)
grid.update_topography(dt)

diagnostics = dv(state)
state.update(diagnostics)

tendencies, _ = pg(state)
u_tnd = tendencies['x_velocity'].values

print('End.')
