from datetime import datetime, timedelta
import gridtools as gt
import numpy as np
from sympl import DataArray


dtype   = np.float64
backend = gt.mode.NUMPY

domain_x = DataArray([-250, 250], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 101
domain_y = DataArray([-1, 1], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 1
domain_z = DataArray([340, 280], dims='potential_temperature', attrs={'units': 'K'})
nz       = 60

topo_type   = 'gaussian'
topo_time   = timedelta(seconds=1800)
topo_kwargs = {
    'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
    'topo_width_x': DataArray(25.0, attrs={'units': 'km'}),
	'topo_smooth': False,
}

init_time	 	   = datetime(year=1992, month=2, day=20)
init_x_velocity    = DataArray(15.0, attrs={'units': 'm s^-1'})
init_y_velocity    = DataArray(0.0, attrs={'units': 'm s^-1'})
init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

time_integration_scheme  = 'forward_euler'
horizontal_flux_scheme   = 'upwind'
horizontal_boundary_type = 'relaxed'

damp_on             = False
damp_type           = 'rayleigh'
damp_depth          = 20
damp_max            = 0.05
damp_at_every_stage = True

smooth_on             = True
smooth_type           = 'first_order'
smooth_damp_depth     = 30
smooth_coeff          = 0.2
smooth_coeff_max      = 1.0
smooth_at_every_stage = True

smooth_moist_on             = False
smooth_moist_type           = 'third_order'
smooth_moist_damp_depth     = 30
smooth_moist_coeff          = 0.2
smooth_moist_coeff_max      = 1.0
smooth_moist_at_every_stage = True

sedimentation    = False
rain_evaporation = False

timestep = timedelta(seconds=10)
niter    = int(21600 / timestep.total_seconds())

filename        = '../tests/baseline_datasets/isentropic_moist.nc'
save_frequency  = int(niter/2)
print_frequency = 60
