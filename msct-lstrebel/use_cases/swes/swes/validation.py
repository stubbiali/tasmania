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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains
"""
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import os

import argparse



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



def calculate_difference(ref_field, target_field):
    diff = np.subtract(ref_field, target_field)
    abs_diff = np.absolute(diff)
    abs_err_sum = np.sum(abs_diff)
    abs_err_avg = np.mean(abs_diff)

    # Relative error calculation with formula from:
    # https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
    rel_diff = np.absolute(np.subtract(ref_field, target_field)) * 2.0
    with np.errstate(invalid='ignore'):
        # rel_diff = np.divide(rel_diff, np.maximum(np.absolute(ref_field), np.absolute(target_field)))
        rel_diff = np.divide(rel_diff, np.absolute(ref_field) + np.absolute(target_field))
    rel_diff[np.maximum(np.absolute(ref_field), np.absolute(target_field)) == 0.0] = 0.0

    rel_err_sum = np.sum(rel_diff)
    rel_err_avg = np.mean(rel_diff)

    return diff, abs_err_sum, abs_err_avg, rel_err_sum, rel_err_avg




def compare_files(ref_file, target_file, fileoutput=False, path="", prefix="", plotting=False, onlylast=False):
    # Load data
    f_ref = np.load(ref_file)
    t_ref, phi_ref, theta_ref, h_ref = f_ref['t'], f_ref['phi'], f_ref['theta'], f_ref['h']

    f_tar = np.load(target_file)
    t_tar, phi_tar, theta_tar, h_tar = f_tar['t'], f_tar['phi'], f_tar['theta'], f_tar['h']


    h_abs_err_sum = np.zeros(len(t_tar))
    h_abs_err_avg = np.zeros(len(t_tar))
    h_rel_err_sum = np.zeros(len(t_tar))
    h_rel_err_avg = np.zeros(len(t_tar))


    if onlylast:
        start_time = len(t_tar) - 1
    else:
        h_diff_per_cell_sum = np.zeros_like(h_tar[:, :, 0])
        start_time = 0

    for t in range(start_time, len(t_tar)):
        h_diff, h_abs_err_sum[t], h_abs_err_avg[t], h_rel_err_sum[t], h_rel_err_avg[t] = calculate_difference(
            h_ref[:, :, t], h_tar[:, :, t])

        # if plotting and t > start_time:
        #         plot_difference_field(t_tar, h_diff, phi_tar, theta_tar,
        #                               filename=str(path) + str(prefix) + "_" + str(t) + "_" + "ref_vs_dd", ct=t)

        if not onlylast:
            h_diff_per_cell_sum[:] += h_diff[:]

        # if plotting:
        #     plot_difference_field(t_tar, h_diff_per_cell_sum, phi_tar, theta_tar, filename=str(path) + str(prefix)
        #                                                                                   + "_" + str(t) + "_" + "ref_vs_dd")

    if fileoutput:
        save_differences_to_file([h_abs_err_avg, h_rel_err_avg],
                                 str(path) + str(prefix) + "ref_vs_dd.csv")

    if plotting:
        if onlylast:
            plot_difference_field(t_tar, h_diff, phi_tar, theta_tar, filename=str(path) + str(prefix) + "ref_vs_dd")
        else:
            h_diff_per_cell_sum[:] /= len(t_tar)
            plot_difference_field(t_tar, h_diff_per_cell_sum, phi_tar, theta_tar, filename=str(path) + str(prefix)
                                                                                           + "ref_vs_dd_time_avg")


def plot_difference_field(t, h, phi, theta, filename, ct=-1):
    plt.cla()
    plt.clf()

    projection = 'plate_carree'
    # Tunable settings
    fontsize = 14
    figsize = [7, 8]
    color_scale_levels = 29
    cmap_name = 'BuRd'
    cbar_orientation = 'horizontal'
    cbar_ticks_pos = 'edge'
    cbar_ticks_step = 4

    plt.rcParams['font.size'] = fontsize

    fig = plt.figure(figsize=figsize)

    proj = get_projection(projection)
    x, y = phi * 180.0 / math.pi, theta * 180.0 / math.pi

    ax = plt.axes(projection=proj)
    ax.coastlines()
    # ax.gridlines()
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

    surf = plt.contourf(x, y, h[:, :], color_scale, transform=proj, cmap=cm)

    plt.title('Fluid height [m]', loc='left', fontsize=fontsize - 1)
    # plt.title(get_time_string(t[n][0]), loc='right', fontsize=fontsize-1)
    plt.title(get_time_string(t[ct]), loc='right', fontsize=fontsize - 1)

    cb = plt.colorbar(surf, orientation=cbar_orientation)

    if cbar_ticks_pos == 'center':
        cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
    else:
        cb.set_ticks(color_scale[::cbar_ticks_step])

    fig.tight_layout()
    # plt.show()
    plt.savefig(str(filename) + ".png")


def save_differences_to_file(errors, filename):
    save_arr = np.column_stack(errors)
    np.savetxt(filename, save_arr, delimiter=",", fmt="%10.7E")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run difference calculations, save them to csv "
                                                 "or plot them for validation purposes.")

    parser.add_argument("f", default=None, type=str,
                        help="Name of the target .npz file.")
    parser.add_argument("-r", default=None, type=str,
                        help="Optional: Name of the reference .npz file.")
    parser.add_argument("-o", action="store_true",
                        help="Optional: Store differences in csv file.")
    parser.add_argument("-loc", default="", type=str,
                        help="Path to location where files should be saved to.")
    parser.add_argument("-pf", default="test_", type=str,
                        help="Optional: Prefix for file names.")
    parser.add_argument("-p", action="store_true",
                        help="Optional: Plot difference map.")

    args = parser.parse_args()

    if args.f is None:
        raise ValueError("No pickle file to compare given.")


    compare_files(args.r, args.f, fileoutput=args.o, path=args.loc, prefix=args.pf, plotting=args.p)

