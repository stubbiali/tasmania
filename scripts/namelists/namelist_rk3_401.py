from datetime import datetime, timedelta
import gridtools as gt
import numpy as np
from sympl import DataArray


dtype   = np.float64
backend = gt.mode.NUMPY

domain_x = DataArray([-220, 220], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 401
domain_y = DataArray([-1, 1], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 1
domain_z = DataArray([765, 300], dims='potential_temperature', attrs={'units': 'K'})
nz       = 300

topo_type   = 'user_defined'
topo_time   = timedelta(seconds=900)
topo_kwargs = {
	'topo_str': '1 * 10000. * 10000. / (x*x + 10000.*10000.)',
	'topo_smooth': False,
}

init_time	 = datetime(year=1992, month=2, day=20)
init_x_velocity  = DataArray(10.0, attrs={'units': 'm s^-1'})
init_y_velocity  = DataArray(0.0, attrs={'units': 'm s^-1'})
init_temperature = DataArray(250.0, attrs={'units': 'K'})

time_integration_scheme  = 'rk3'
horizontal_flux_scheme   = 'fifth_order_upwind'
horizontal_boundary_type = 'relaxed'

damp_on             = True
damp_type           = 'rayleigh'
damp_depth          = 150
damp_max            = 0.05
damp_at_every_stage = False

smooth_on             = False
smooth_type           = 'first'
smooth_coeff          = 0.05
smooth_at_every_stage = False

timestep = timedelta(seconds=2.5)
niter    = int(120000 / timestep.total_seconds())

filename        = '../data/isentropic_convergence_{}_{}_nx{}_dt{}_nt{}.nc'.format(
					time_integration_scheme, horizontal_flux_scheme, nx, int(timestep.total_seconds()), niter)
save_frequency  = int(niter/2)
print_frequency = 100