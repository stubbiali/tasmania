import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import os
import pickle
import scipy.io


def _projection_factory(name):
	if name == 'plate_carree':
		return ccrs.PlateCarree()
	elif name == 'mercator':
		return ccrs.Mercator()
	elif name == 'miller':
		return ccrs.Miller()
	elif name == 'mollweide':
		return ccrs.Mollweide()
	elif name == 'orthographic':
		return ccrs.Orthographic()
		

def ess_movie_maker(file_name, to_plot, projection = 'plate_carree', 
			 movie_format = 'mpg', movie_fps = 15, movie_name = None):
	"""
	Plot Earth system science data.
	This utility leverage on CartoPy and matplotlib packages.

	:param file_name	Path to Python dataset(s) storing the results
	:param to_plot		What to plot
						* 'h': fluid height
						* 'u': longitudinal velocity
						* 'v': latitudinal velocity
						* 'quiver': velocity quiver-plot
						* 'vorticity': relative vorticity
						* 'mesh': underlying computational grid
	:param projection	Projection
						* 'plate_carree' (default)
						* 'mercator'
						* 'miller'
						* 'mollweide'
						* 'orthographic'
	:param movie_format Format for animation
						* 'mpg' (default)
						* 'mp4'
	:param movie_fps	Frames-per-second; default is 15
	:param movie_name	Movie name, without extension
	"""
	#
	# Load data 
	#
	if to_plot == 'h':
		with open(file_name, 'rb') as file_:
			t, phi, theta, h = pickle.load(file_)
	elif to_plot == 'mesh':
		with open(file_name, 'rb') as file_:
			_, phi, theta, _ = pickle.load(file_)
	else:
		with open(file_name[0], 'rb') as file_:
			t, phi, theta, u = pickle.load(file_)
		with open(file_name[1], 'rb') as file_:
			_, _, _, v = pickle.load(file_)

	# 
	# Plot height
	#
	if to_plot == 'h':
		n, nt, fig = 0, h.shape[2], plt.figure(figsize=[15,8])
		
		# Set movie name
		root, _ = os.path.splitext(file_name)
		if movie_name is None:
			movie_name = root + '.' + movie_format
		else:
			movie_name += ('.' + movie_format)
		
		# Instantiate writer class
		ffmpeg_writer = manimation.writers["ffmpeg"]
		metadata = dict(title = 'Fluid height')
		writer = ffmpeg_writer(fps = movie_fps, metadata = metadata)
		
		with writer.saving(fig, movie_name, nt):
			proj = _projection_factory(projection)
			x, y = phi*180.0/math.pi, theta*180.0/math.pi
				
			ax = plt.axes(projection = proj)
			ax.coastlines()
			#ax.gridlines()
			ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs = proj)
			ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs = proj)
			lon_formatter = LongitudeFormatter(zero_direction_label = False,
                            number_format='.0f')
			lat_formatter = LatitudeFormatter()
			ax.xaxis.set_major_formatter(lon_formatter)
			ax.yaxis.set_major_formatter(lat_formatter)

			for n in range(nt):
				if n > 0:
					for coll in surf.collections:
						plt.gca().collections.remove(coll) 
			
				color_scale = np.linspace(h.min(), h.max(), 30, endpoint = True)
				surf = plt.contourf(x, y, h[:,:,n], color_scale, transform = proj) # Optional arguments: cmap, extend
				#ax.pcolormesh(x, y, h[1:-1,:,n], transform = proj)

				plt.title('Fluid height [m], %5.2f hours\n' % (t[n] / 3600.))
				
				# Uncomment the following lines to show the colorbar on the left-hand side of the plot
				if (n == 0):
					plt.colorbar(surf, orientation = 'horizontal')
					
				writer.grab_frame()
	
		plt.show()
	
	#
	# Plot longitudinal velocity
	#
	if to_plot == 'u':
		pass

	#
	# Plot longitudinal velocity
	#
	if to_plot == 'u':
		pass
	
	#
	# Plot latitudinal velocity
	#
	if to_plot == 'v':
		pass

	#
	# Plot velocity quiver-plot
	#
	if to_plot == 'quiver':
		pass

	#
	# Plot relative vorticity
	#
	if to_plot == 'vorticity':
		pass

	#
	# Plot computational grid
	#
	if to_plot == 'mesh':
		pass
