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
import numpy as np
import tasmania as taz
import matplotlib.pyplot as plt

# ==================================================
# User inputs
# ==================================================
values = [
    [
        93438.55499049266,
        92546.30188995667,
        92741.48574232402,
        92522.62088689643,
        95438.1293806084,
    ],
    [
        88990.91027228518,
        88584.96736412216,
        88766.59838146689,
        88573.9212174644,
        90028.40318340881,
    ],
    [
        87343.51949616159,
        87151.06663923517,
        87251.37040591387,
        87144.49161664736,
        87867.70894757634,
    ]
]

colors = ['darkviolet', 'royalblue', 'gold']

edgecolors = ['black', ]*3

labels = ['$\\Delta x = 8.8$ km', '$\\Delta x = 4.4$ km', '$\\Delta x = 2.2$ km']

bar_width = 0.25

figure_properties = {
    'fontsize': 16,
    'figsize': (6, 7),
    'tight_layout': True,
    'tight_layout_rect': None,
}

axes_properties = {
    'fontsize': 16,
    'x_label': '',
    'x_labelcolor': 'black',
    'x_lim': None,
    'invert_xaxis': False,
    'x_scale': None,
    'x_ticks': [0.25, 1.25, 2.25, 3.25, 4.25],
    'x_ticklabels': ['CC', 'LCC', 'PS', 'SUS', 'SSUS'],
    'x_tickcolor': 'black',
    'xaxis_minor_ticks_visible': False,
    'xaxis_visible': True,
    # y-axis
    'y_label': 'Accumulated precipitation [10$^9$ kg]',
    'y_labelcolor': 'black',
    'y_lim': [8e4, 10e4],
    'invert_yaxis': False,
    'y_scale': None,
    'y_ticks': [8e4, 8.5e4, 9e4, 9.5e4, 10e4],
    'y_ticklabels': [80, 85, 90, 95, 100],
    'y_tickcolor': 'black',
    'yaxis_minor_ticks_visible': False,
    'yaxis_visible': True,
    # z-axis
    'z_label': '',
    'z_labelcolor': '',
    'z_lim': None,
    'invert_zaxis': False,
    'z_scale': None,
    'z_ticks': None,
    'z_ticklabels': None,
    'z_tickcolor': 'white',
    'zaxis_minor_ticks_visible': True,
    'zaxis_visible': True,
    # legend
    'legend_on': True,
    'legend_loc': 'best', #'center left',
    'legend_bbox_to_anchor': None, #(1.04, 0.5),
    'legend_framealpha': 1.0,
    'legend_ncol': 1,
    # textbox
    'text': None, #'$w_{\\mathtt{FW}} = 0$',
    'text_loc': 'upper left',
    # grid
    'grid_on': True,
    'grid_properties': {'linestyle': ':'},
}

# ==================================================
# Code
# ==================================================
x = range(len(values[0]))

fig, ax = taz.get_figure_and_axes(**figure_properties)

for i in range(len(values)):
    ax.bar(
        [elx + i*bar_width for elx in x], values[i], bar_width, 
        color=colors[i], edgecolor=edgecolors[i], label=labels[i]
    )

taz.set_axes_properties(ax, **axes_properties)
taz.set_figure_properties(fig, **figure_properties)

plt.show()

