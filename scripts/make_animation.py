import tasmania as taz


#
# User inputs
#
module = 'make_contourf_xz'

tlevels = range(0, 10, 2)

print_time = 'elapsed'  # 'elapsed', 'absolute'
fps = 10

save_dest = '../results/movies/smolarkiewicz/rk2_third_order_upwind_centered_' \
			'nx51_ny51_nz50_dt20_nt8640_flat_terrain.mp4'


#
# Code
#
if __name__ == '__main__':
	exec('from {} import get_plot as get_artist, get_states'.format(module))
	artist = locals()['get_artist']()

	engine = taz.Animation(artist, print_time=print_time, fps=fps)

	for t in tlevels:
		engine.store(locals()['get_states'](t, artist))

	engine.run(save_dest=save_dest)
