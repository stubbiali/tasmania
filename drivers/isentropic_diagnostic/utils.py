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
from tasmania.python.framework.generic_functions import to_numpy


def print_info(dt, i, nl, pgrid, state):
    if (nl.print_dry_frequency > 0) and (
        (i + 1) % nl.print_dry_frequency == 0
    ):
        s = to_numpy(
            state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
        )
        su = to_numpy(
            state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
        )
        sv = to_numpy(
            state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
        )

        u = su[3:-4, 3:-4, :-1] / s[3:-4, 3:-4, :-1]
        v = sv[3:-4, 3:-4, :-1] / s[3:-4, 3:-4, :-1]
        umax, umin = u.max(), u.min()
        vmax, vmin = v.max(), v.min()
        cfl = max(
            umax * dt.total_seconds() / pgrid.dx.to_units("m").values.item(),
            vmax * dt.total_seconds() / pgrid.dy.to_units("m").values.item(),
        )

        # print useful info
        print(
            f"Iteration {i+1:6d}: CFL = {cfl:4f}, "
            f"umax = {umax:10.8f} m/s, umin = {umin:10.8f} m/s, "
            f"vmax = {vmax:10.8f} m/s, vmin = {vmin:10.8f} m/s"
        )

    if (nl.print_moist_frequency > 0) and (
        (i + 1) % nl.print_moist_frequency == 0
    ):
        qv = to_numpy(
            state["mass_fraction_of_water_vapor_in_air"]
            .to_units("g g^-1")
            .data
        )
        qv_max = qv[3:-4, 3:-4, :-1].max() * 1e3
        qc = to_numpy(
            state["mass_fraction_of_cloud_liquid_water_in_air"]
            .to_units("g g^-1")
            .data
        )
        qc_max = qc[3:-4, 3:-4, :-1].max() * 1e3
        qr = to_numpy(
            state["mass_fraction_of_precipitation_water_in_air"]
            .to_units("g g^-1")
            .data
        )
        qr_max = qr[3:-4, 3:-4, :-1].max() * 1e3

        if "precipitation" in state:
            prec = to_numpy(state["precipitation"].to_units("mm hr^-1").data)
            prec_max = prec[3:-4, 3:-4].max()
            accprec = to_numpy(
                state["accumulated_precipitation"].to_units("mm").data
            )
            accprec_max = accprec[3:-4, 3:-4].max()

            print(
                f"Iteration {i+1:6d}: "
                f"qvmax = {qv_max:10.8f} g/kg, "
                f"qcmax = {qc_max:10.8f} g/kg, "
                f"qrmax = {qr_max:10.8f} g/kg, "
                f"prec_max = {prec_max:10.8f} mm/hr, "
                f"accprec_max = {accprec_max:10.8f} mm"
            )
        else:
            print(
                f"Iteration {i+1:6d}: "
                f"qvmax = {qv_max:10.8f} g/kg, "
                f"qcmax = {qc_max:10.8f} g/kg, "
                f"qrmax = {qr_max:10.8f} g/kg"
            )
