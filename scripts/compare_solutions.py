import numpy as np
import tasmania as taz


#
# User inputs
#
filename1 = '../data/smolarkiewicz_rk2_third_order_upwind_nx51_ny51_nz50_dt20_nt6480.nc'
filename2 = '../data/smolarkiewicz_rk2_third_order_upwind_nx51_ny51_nz50_dt20_nt6480_bis.nc'

tlevels1 = range(0, 72)
tlevels2 = range(0, 72)

fieldname1 = 'x_velocity_at_u_locations'
fieldname2 = 'x_velocity_at_u_locations'

units1 = 'm s^-1'
units2 = 'm s^-1'

#
# Code
#
if __name__ == '__main__':
	grid1, states1 = taz.load_netcdf_dataset(filename1)
	grid2, states2 = taz.load_netcdf_dataset(filename2)

	for t1, t2 in zip(tlevels1, tlevels2):
		field1 = states1[t1][fieldname1].to_units(units1).values
		field2 = states2[t2][fieldname2].to_units(units2).values

		isclose = np.allclose(field1, field2)

		if isclose:
			print('Iteration ({:4d}, {:4d}) validated.'.format(t1, t2))
		else:
			print('Iteration ({:4d}, {:4d}) not validated.'.format(t1, t2))
