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

import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

import utils


def calculate_difference(ref_field, target_field):
    diff = np.subtract(ref_field, target_field)
    abs_diff = np.absolute(diff)
    abs_err_sum = np.sum(abs_diff)
    abs_err_avg = np.mean(abs_diff)


    # Relative error calculation with formula from:
    # https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
    rel_diff = np.subtract(ref_field, target_field)
    rel_diff = 2.0 * rel_diff #* np.absolute(rel_diff)
    with np.errstate(invalid='ignore'):
        # rel_diff = np.divide(rel_diff, np.maximum(np.absolute(ref_field), np.absolute(target_field)))
        rel_diff = np.divide(rel_diff, np.absolute(ref_field) + np.absolute(target_field))
    # rel_diff[np.maximum(np.absolute(ref_field), np.absolute(target_field)) == 0.0] = 0.0

    rel_err_sum = np.sum(rel_diff)
    rel_err_avg = np.mean(rel_diff)

    return diff, abs_err_sum, abs_err_avg, rel_err_sum, rel_err_avg


def compare_zhao_exact_with_file(target_file, fileoutput=False, path="", prefix="", plotting=False, onlylast=False,
                                 extent=(0, 1, 0, 1)):
    with open(target_file, 'rb') as data:
        t2 = pickle.load(data)
        x2 = pickle.load(data)
        y2 = pickle.load(data)
        u2 = pickle.load(data)
        v2 = pickle.load(data)
        eps2 = pickle.load(data)

    u_abs_err_sum = np.zeros(len(t2))
    u_abs_err_avg = np.zeros(len(t2))
    u_rel_err_sum = np.zeros(len(t2))
    u_rel_err_avg = np.zeros(len(t2))

    v_abs_err_sum = np.zeros(len(t2))
    v_abs_err_avg = np.zeros(len(t2))
    v_rel_err_sum = np.zeros(len(t2))
    v_rel_err_avg = np.zeros(len(t2))

    u1 = np.zeros_like(u2)
    v1 = np.zeros_like(v2)

    if onlylast:
        start_time = len(t2) - 1
    else:
        u_diff_per_cell_sum = np.zeros_like(u2[:, :, 0])
        v_diff_per_cell_sum = np.zeros_like(v2[:, :, 0])
        start_time = 0

    for t in range(start_time, len(t2)):
        t_ = t2[t].seconds if t2[t].seconds > 0 else t2[t].microseconds / 1.e6
        u1[:, :, t] = (- 2. * eps2 * 2. * np.pi * np.exp(- 5. * np.pi * np.pi * eps2 * t_) * np.cos(2. * np.pi * x2)
                       * np.sin(np.pi * y2) / (2. + np.exp(- 5. * np.pi * np.pi * eps2 * t_) * np.sin(2. * np.pi * x2)
                                               * np.sin(np.pi * x2)))
        v1[:, :, t] = (- 2. * eps2 * np.pi * np.exp(- 5. * np.pi * np.pi * eps2 * t_) * np.sin(2. * np.pi * x2)
                       * np.cos(np.pi * y2) / (2. + np.exp(- 5. * np.pi * np.pi * eps2 * t_) * np.sin(2. * np.pi * x2)
                                               * np.sin(np.pi * x2)))

        u_diff, u_abs_err_sum[t], u_abs_err_avg[t], u_rel_err_sum[t], u_rel_err_avg[t] = calculate_difference(
            u1[:, :, t], u2[:, :, t])

        v_diff, v_abs_err_sum[t], v_abs_err_avg[t], v_rel_err_sum[t], v_rel_err_avg[t] = calculate_difference(
            v1[:, :, t], v2[:, :, t])

        if not onlylast:
            u_diff_per_cell_sum[:] += u_diff[:]
            v_diff_per_cell_sum[:] += v_diff[:]

    if fileoutput:
        save_differences_to_file([u_abs_err_avg, u_rel_err_avg, v_abs_err_avg, v_rel_err_avg],
                                 str(path) + str(prefix) + "zhao_exact_vs_dd.csv")

    if plotting:
        if onlylast:
            plot_difference_field(u_diff, v_diff, extent=extent, filename=str(path) + str(prefix) + "zhao_exact_vs_dd")
        else:
            u_diff_per_cell_sum[:] /= len(t2)
            plot_difference_field(u_diff_per_cell_sum, v_diff_per_cell_sum, extent=extent,
                                  filename=str(path) + str(prefix) + "zhao_exact_vs_dd_time_avg")


def compare_pickle_files(ref_file, target_file, fileoutput=False, path="", prefix="", plotting=False, onlylast=False,
                         extent=(0, 1, 0, 1)):
    # Load data
    with open(ref_file, 'rb') as data:
        t1 = pickle.load(data)
        x1 = pickle.load(data)
        y1 = pickle.load(data)
        u1 = pickle.load(data)
        v1 = pickle.load(data)
        eps1 = pickle.load(data)

    with open(target_file, 'rb') as data:
        t2 = pickle.load(data)
        x2 = pickle.load(data)
        y2 = pickle.load(data)
        u2 = pickle.load(data)
        v2 = pickle.load(data)
        eps2 = pickle.load(data)

    u_abs_err_sum = np.zeros(len(t2))
    u_abs_err_avg = np.zeros(len(t2))
    u_rel_err_sum = np.zeros(len(t2))
    u_rel_err_avg = np.zeros(len(t2))

    v_abs_err_sum = np.zeros(len(t2))
    v_abs_err_avg = np.zeros(len(t2))
    v_rel_err_sum = np.zeros(len(t2))
    v_rel_err_avg = np.zeros(len(t2))

    if onlylast:
        start_time = len(t2) - 1
    else:
        u_diff_per_cell_sum = np.zeros_like(u2[:, :, 0])
        v_diff_per_cell_sum = np.zeros_like(v2[:, :, 0])
        start_time = 0

    for t in range(start_time, len(t2)):
        u_diff, u_abs_err_sum[t], u_abs_err_avg[t], u_rel_err_sum[t], u_rel_err_avg[t] = calculate_difference(
            u1[:, :, t], u2[:, :, t])

        v_diff, v_abs_err_sum[t], v_abs_err_avg[t], v_rel_err_sum[t], v_rel_err_avg[t] = calculate_difference(
            v1[:, :, t], v2[:, :, t])

        if not onlylast:
            u_diff_per_cell_sum[:] += u_diff[:]
            v_diff_per_cell_sum[:] += v_diff[:]

    if fileoutput:
        save_differences_to_file([u_abs_err_avg, u_rel_err_avg, v_abs_err_avg, v_rel_err_avg],
                                 str(path) + str(prefix) + "ref_vs_dd.csv")

    if plotting:
        if onlylast:
            plot_difference_field(u_diff, v_diff, extent=extent, filename=str(path) + str(prefix) + "zhao_exact_vs_dd")
        else:
            u_diff_per_cell_sum[:] /= len(t2)
            plot_difference_field(u_diff_per_cell_sum, v_diff_per_cell_sum, extent=extent,
                                  filename=str(path) + str(prefix) + "zhao_exact_vs_dd_time_avg")

def plot_difference_field(field1, field2, extent, filename):
    cm = utils.reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')

    surf = plt.imshow(field1.T, extent=extent, origin='lower', cmap=cm, interpolation='none')
    cb = plt.colorbar(orientation="horizontal", fraction=0.046)
    # plt.show()
    plt.savefig(str(filename) + "_field1.png")
    plt.clf()
    plt.cla()

    surf = plt.imshow(field2.T, extent=extent, origin='lower', cmap=cm, interpolation='none')
    cb = plt.colorbar(orientation="horizontal", fraction=0.046)
    plt.savefig(str(filename) + "_field2.png")


def save_differences_to_file(errors, filename):
    save_arr = np.column_stack(errors)
    np.savetxt(filename, save_arr, delimiter=",", fmt="%10.7E")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run difference calculations, save them to csv "
                                                 "or plot them for validation purposes.")

    parser.add_argument("f", default=None, type=str,
                        help="Name of the target pickle file.")

    parser.add_argument("-r", default=None, type=str,
                        help="Optional: Name of the reference pickle file.")

    parser.add_argument("-o", action="store_true",
                        help="Optional: Store differences in csv file.")
    parser.add_argument("-loc", default="", type=str,
                        help="Path to location where files should be saved to.")
    parser.add_argument("-pf", default="test_", type=str,
                        help="Optional: Prefix for file names.")
    parser.add_argument("-p", action="store_true",
                        help="Optional: Plot difference map.")
    parser.add_argument('-e', default=(0, 1, 0, 1), nargs='+', type=int,
                        help="Optional: Plotting extent [4 int].")

    args = parser.parse_args()

    if args.f is None:
        raise ValueError("No pickle file to compare given.")

    if args.r is None:
        compare_zhao_exact_with_file(args.f, fileoutput=args.o, path=args.loc, prefix=args.pf,
                                     plotting=args.p, extent=tuple(args.e))
    else:
        compare_pickle_files(args.r, args.f, fileoutput=args.o, path=args.loc, prefix=args.pf,
                             plotting=args.p, extent=tuple(args.e))

