# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import os


def get_projection(name):
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


def reverse_colormap(cmap, name=None):
	"""
	Reverse a Matplotlib colormap.

	Parameters
	----------
	cmap : obj
		The :class:`matplotlib.colors.LinearSegmentedColormap` to invert.
	name : `str`, optional
		The name of the reversed colormap. By default, this is obtained by appending '_r'
		to the name of the input colormap.

	Return
	------
	obj :
		The reversed :class:`matplotlib.colors.LinearSegmentedColormap`.

	References
	----------
	https://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib.
	"""
	keys = []
	reverse = []

	for key in cmap._segmentdata:
		# Extract the channel
		keys.append(key)
		channel = cmap._segmentdata[key]

		# Reverse the channel
		data = []
		for t in channel:
			data.append((1-t[0], t[2], t[1]))
		reverse.append(sorted(data))

	# Set the name for the reversed map
	if name is None:
		name = cmap.name + '_r'

	return LinearSegmentedColormap(name, dict(zip(keys, reverse)))


def get_time_string(time_in_seconds):
	days = int(time_in_seconds / (24*60*60))
	time_in_seconds -= days*24*60*60

	hours = int(time_in_seconds / (60*60))
	time_in_seconds -= hours*60*60

	minutes = int(time_in_seconds / 60)
	seconds = int(time_in_seconds - minutes*60)

	return '%i:%02i:%02i:%02i' % (days, hours, minutes, seconds)


def ess_movie_maker(filename, field_to_plot, projection='plate_carree',
					movie_format='mp4', movie_fps=8, movie_name=None):
	"""
	Utility to generate animations showing the time evolution of
	Earth system science data.
	This utility leverages CartoPy and matplotlib packages.

	Parameters
	----------
	filename : str
		Path to the .npz dataset storing the results.
	field_to_plot : str
		String specifying what should be visualized.
		Available options are:

			* 'h', for the fluid height;
			* 'u', for the longitudinal velocity;
			* 'v', for the latitudinal velocity;
			* 'quiver', for the velocity quiver-plot;
			* 'vorticity', for the relative vorticity;
			* 'mesh', for the underlying computational grid.

	projection : `str`, optional
		String specifying the projection map to use.
		Available options are:

			* 'plate_carree';
			* 'mercator';
			* 'miller';
			* 'mollweide';
			* 'orthographic'.

		Defaults to 'plate_carree'.
	movie_format : `str`, optional
		Format for the output movie. Available options are:

			* 'mpg';
			* 'mp4'.

		Defaults to 'mp4'.
	movie_fps : `int`, optional
		Frames-per-second. Defaults to 10.
	movie_name : `str`, optional
		String denoting the movie name, without extension.
	"""
	#
	# Load data 
	#
	f = np.load(filename)
	if field_to_plot == 'h':
		t, phi, theta, h = f['t'], f['phi'], f['theta'], f['h']
	elif field_to_plot == 'mesh':
		phi, theta = f['phi'], f['theta']
	else:
		t, phi, theta, u, v = f['t'], f['phi'], f['theta'], f['u'], f['v']

	# 
	# Plot height
	#
	if field_to_plot == 'h':
		# Tunable settings
		fontsize = 16
		figsize = [7, 8]
		color_scale_levels = 29
		cmap_name = 'BuRd'
		cbar_orientation = 'horizontal'
		cbar_ticks_pos = 'edge'
		cbar_ticks_step = 4

		plt.rcParams['font.size'] = fontsize

		n, nt, fig = 0, h.shape[2], plt.figure(figsize=figsize)
		
		# Set movie name
		root, _ = os.path.splitext(filename)
		if movie_name is None:
			movie_name = root + '.' + movie_format
		else:
			movie_name += ('.' + movie_format)
		
		# Instantiate writer class
		ffmpeg_writer = manimation.writers["ffmpeg"]
		metadata = dict(title='Fluid height')
		writer = ffmpeg_writer(fps=movie_fps, metadata=metadata)
		
		with writer.saving(fig, movie_name, nt):
			proj = get_projection(projection)
			x, y = phi*180.0/math.pi, theta*180.0/math.pi
				
			ax = plt.axes(projection = proj)
			ax.coastlines()
			#ax.gridlines()
			ax.set_xlim([-180, 180])
			ax.set_ylim([-90, 90])
			ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=proj)
			ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=proj)
			lon_formatter = LongitudeFormatter(zero_direction_label=False,
											   number_format='.0f')
			lat_formatter = LatitudeFormatter()
			ax.xaxis.set_major_formatter(lon_formatter)
			ax.yaxis.set_major_formatter(lat_formatter)

			color_scale = np.linspace(h.min(), h.max(), color_scale_levels, endpoint=True)
			if cmap_name == 'BuRd':
				cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
			else:
				cm = plt.get_cmap(cmap_name)
			# print(h[:, :, 0])
			for n in range(nt):
				if n > 0:
					for coll in surf.collections:
						plt.gca().collections.remove(coll) 

				surf = plt.contourf(x, y, h[:, :, n], color_scale, transform=proj, cmap=cm)

				plt.title('Fluid height [m]', loc='left', fontsize=fontsize-1)
				# plt.title(get_time_string(t[n][0]), loc='right', fontsize=fontsize-1)
				plt.title(get_time_string(t[n]), loc='right', fontsize=fontsize-1)
				
				# Uncomment the following lines to show the colorbar
				if n == 0:
					cb = plt.colorbar(surf, orientation=cbar_orientation)

				if cbar_ticks_pos == 'center':
					cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
				else:
					cb.set_ticks(color_scale[::cbar_ticks_step])

				# fig.tight_layout()
					
				writer.grab_frame()
	
		plt.show()
	
	#
	# Plot longitudinal velocity
	#
	if field_to_plot == 'u':
		pass

	#
	# Plot longitudinal velocity
	#
	if field_to_plot == 'u':
		pass
	
	#
	# Plot latitudinal velocity
	#
	if field_to_plot == 'v':
		pass

	#
	# Plot velocity quiver-plot
	#
	if field_to_plot == 'quiver':
		pass

	#
	# Plot relative vorticity
	#
	if field_to_plot == 'vorticity':
		pass

	#
	# Plot computational grid
	#
	if field_to_plot == 'mesh':
		pass
