import pickle

filename = os.path.join(os.environ['TASMANIA_ROOT'], 
						'data/datasets/isentropic_convergence_upwind_u10_lx400_nz300_8km_relaxed.pickle')
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	u1 = state_save['x_momentum_isentropic'].values[:, 0, :, -1] / state_save['isentropic_density'].values[:, 0, :, -1]

	grid = state_save.grid
	uex, wex = utils.get_isothermal_solution(grid, 10., 250., 1., 1.e4, x_staggered = False, z_staggered = False)

filename = os.path.join(os.environ['TASMANIA_ROOT'], 
						'data/datasets/isentropic_convergence_upwind_u10_lx400_nz300_4km_relaxed.pickle')
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	u2 = state_save['x_momentum_isentropic'].values[:, 0, :, -1] / state_save['isentropic_density'].values[:, 0, :, -1]

print('Done.')
