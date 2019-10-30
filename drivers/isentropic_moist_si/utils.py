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
def print_info(dt, i, nl, pgrid, state):
    if (nl.print_dry_frequency > 0) and ((i + 1) % nl.print_dry_frequency == 0):
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

    if (nl.print_moist_frequency > 0) and ((i + 1) % nl.print_moist_frequency == 0):
        qv_max = (
            state["mass_fraction_of_water_vapor_in_air"]
            .values[10:-11, 10:-11, 30:-1]
            .max()
            .item()
            * 1e3
        )
        qc_max = (
            state["mass_fraction_of_cloud_liquid_water_in_air"]
            .values[10:-11, 10:-11, 30:-1]
            .max()
            .item()
            * 1e3
        )
        qr_max = (
            state["mass_fraction_of_precipitation_water_in_air"]
            .values[10:-11, 10:-11, 30:-1]
            .max()
            .item()
            * 1e3
        )
        if "precipitation" in state:
            prec_max = (
                state["precipitation"].to_units("mm hr^-1").values[10:-10, 10:-10].max()
            )
            accprec_max = (
                state["accumulated_precipitation"]
                .to_units("mm")
                .values[10:-10, 10:-10]
                .max()
                .item()
            )
            print(
                "Iteration {:6d}: qvmax = {:10.8f} g/kg, qcmax = {:10.8f} g/kg, "
                "qrmax = {:10.8f} g/kg, prec_max = {:10.8f} mm/hr, accprec_max = {:10.8f} mm".format(
                    i + 1, qv_max, qc_max, qr_max, prec_max, accprec_max
                )
            )
        else:
            print(
                "Iteration {:6d}: qvmax = {:10.8f} g/kg, qcmax = {:10.8f} g/kg, "
                "qrmax = {:10.8f} g/kg".format(i + 1, qv_max, qc_max, qr_max)
            )
