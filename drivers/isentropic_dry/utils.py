# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
def print_info(dt, i, nl, pgrid, state):
    if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
        u = (
            state["x_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-1")
            .values[3:-4, 3:-4, :-1]
            / state["air_isentropic_density"]
            .to_units("kg m^-2 K^-1")
            .values[3:-4, 3:-4, :-1]
        )
        v = (
            state["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-1")
            .values[3:-4, 3:-4, :-1]
            / state["air_isentropic_density"]
            .to_units("kg m^-2 K^-1")
            .values[3:-4, 3:-4, :-1]
        )

        umax, umin = u.max().item(), u.min().item()
        vmax, vmin = v.max().item(), v.min().item()
        cfl = max(
            umax * dt.total_seconds() / pgrid.dx.to_units("m").values.item(),
            vmax * dt.total_seconds() / pgrid.dy.to_units("m").values.item(),
        )

        # print useful info
        print(
            "Iteration {:6d}: CFL = {:4f}, umax = {:10.8f} m/s, umin = {:10.8f} m/s, "
            "vmax = {:10.8f} m/s, vmin = {:10.8f} m/s".format(
                i + 1, cfl, umax, umin, vmax, vmin
            )
        )
