"""
A script to generate xz-plots.
"""
import os
import pickle

#
# Mandatory settings
#
filename = os.path.join(os.environ['GT4ESS_ROOT'], 'post_processing/data/verification_1_maccormack.pickle')
field = 'x_velocity'
y_level = 25
time_level = -1

#
# Optional settings
#
title           = '$x$-velocity [$m s^{-1}$]'
x_factor        = 1.e-3
x_label         = 'x [$km$]'
z_factor        = 1.e-3
z_label			= 'z [$km$]'
cmap_name       = 'RdBu' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cmap_levels     = 14
cmap_center     = 15.
cmap_half_width = 3.

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	state_save.plot_xz(field, y_level, time_level, 
					   title           = title, 
					   x_label         = x_label,
					   x_factor        = x_factor,
					   z_label         = z_label,
					   z_factor        = z_factor,
					   cmap_name       = cmap_name, 
					   cmap_levels     = cmap_levels,
					   cmap_center     = cmap_center, 
					   cmap_half_width = cmap_half_width
					  )
