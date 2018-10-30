import tasmania as taz


#
# User inputs
#
module = 'make_contourf_xz'

time_levels = range(0, 10, 2)

fontsize = 16
figsize = (7, 8)
tight_layout = True
print_time = 'elapsed'
fps = 10
save_dest = """../results/movies/smolarkiewicz/
			   rk2_third_order_upwind_centered_nx51_ny51_nz50_dt20_nt8640_flat_terrain.mp4"""


#
# Code
#
if __name__ == '__main__':
	exec('from {} import get_artist, get_loader'.format(module))
	artist, _ = locals()['get_artist'](time_levels[0])
	loader = locals()['get_loader']()

	engine = taz.Animation(artist, fontsize=fontsize, figsize=figsize,
						   tight_layout=tight_layout, print_time=print_time, fps=fps)

	for t in time_levels:
		engine.store(loader(t))

	engine.run(save_dest=save_dest)
