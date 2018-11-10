from datetime import datetime
from loader import LoaderFactory
import tasmania as taz

#
# User inputs
#
filename1 = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx51_ny51_nz50_dt20_nt4320_flat_terrain_L25000_u0_wf4_f_sus_bis.nc'
filename2 = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx51_ny51_nz50_dt20_nt4320_flat_terrain_L25000_u0_wf4_f_cc_bis.nc'

field_name  = 'x_velocity_at_u_locations'
field_units = 'm s^-1'

x1, x2 = None, None
y1, y2 = None, None
z1, z2 = None, None

time_mode     = 'elapsed'
init_time     = datetime(year=1992, month=2, day=20, hour=8)
time_units    = 'hr'
time_on_xaxis = True

drawer_properties = {
	'fontsize': 16,
	'linestyle': '-',
	'linewidth': 1.5,
	'linecolor': 'blue',
	'marker': '^',
	'markersize': 7,
	'markeredgewidth': 1,
	'markerfacecolor': 'white',
	'markeredgecolor': 'blue',
	'legend_label': 'SUS'
}


#
# Code
#
def get_drawer():
	loader1 = LoaderFactory.factory(filename1)
	grid1 = loader1.get_grid()

	drawer = taz.TimeSeries(
		grid1, 'rrmsd_of_' + field_name, None,
		time_mode=time_mode, init_time=init_time,
		time_units=time_units, time_on_xaxis=time_on_xaxis,
		properties=drawer_properties
	)

	return drawer


def get_state(tlevel, drawer, axes_properties=None, print_time=None):
	loader1 = LoaderFactory.factory(filename1)
	loader2 = LoaderFactory.factory(filename2)

	grid1 = loader1.get_grid()
	grid2 = loader2.get_grid()

	rrmsd = taz.RRMSD(
		(grid1, grid2), {field_name: field_units},
		x=(x1, x2), y=(y1, y2), z=(z1, z2)
	)

	drawer.reset()

	tlevel = loader1.nt + tlevel if tlevel < 0 else tlevel

	for k in range(0, tlevel-1):
		state1, state2 = loader1.get_state(k), loader2.get_state(k)
		diagnostics = rrmsd(state1, state2)
		state1.update(diagnostics)
		drawer(state1)

	state1, state2 = loader1.get_state(tlevel), loader2.get_state(tlevel)
	diagnostics = rrmsd(state1, state2)
	state1.update(diagnostics)

	return state1
