"""
A script to generate xy-plots.
"""
import os
import pickle

#
# Mandatory settings
#
filename = os.path.join(os.environ['GT4ESS_ROOT'], 'post_processing/data/verification_1_maccormack.pickle')
field = 'horizontal_velocity'
z_level = -1
time_level = -1

#
# Optional settings
#
title           = '$x$-velocity [$m s^{-1}$]'
x_label         = 'x [$km$]'
x_factor        = 1.e-3
y_label         = 'y [$km$]'
y_factor		= 1.e-3
cmap_name       = 'RdBu' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cmap_levels     = 14
cmap_center     = 15.
cmap_half_width = 3.5

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	state_save.plot_xy(field, z_level, time_level, 
					   title           = title, 
					   x_label         = x_label,
					   x_factor        = x_factor,
					   y_label         = y_label,
					   y_factor        = y_factor,
					   cmap_name       = cmap_name, 
					   cmap_levels     = cmap_levels, 
					   cmap_center     = cmap_center, 
					   cmap_half_width = cmap_half_width,
					  )
