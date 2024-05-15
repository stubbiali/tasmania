# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
def deepcopy_Line2D(line):
    """
    Return a deep copy of a :class:`matplotlib.lines.Line2D` object.

    Parameters
    ----------
    line : obj
            The :class:`matplotlib.lines.Line2D` to copy.

    Return
    ------
    obj :
            A deep copy of the input :class:`matplotlib.lines.Line2D`.
    """
    assert type(line) == Line2D

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    out_line = Line2D(xdata, ydata)
    out_line.update_from(line)

    return out_line
