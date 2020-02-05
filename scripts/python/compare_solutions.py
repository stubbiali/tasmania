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
import argparse
import numpy as np
import tasmania as taz


# ========================================
# user inputs
# ========================================
filename1 = (
    "../../data/isentropic_prognostic-validation/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_"
    "nx41_ny41_nz60_dt40_nt360_gaussian_L50000_H1000_u15_rh90_turb_f_sed_evap_ssus_numpy.nc"
)
filename2 = (
    "../../data/isentropic_prognostic-validation/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_"
    "nx41_ny41_nz60_dt40_nt360_gaussian_L50000_H1000_u15_rh90_turb_f_sed_evap_ssus_gtx86.nc"
)

field_properties = {
    "accumulated_precipitation": {"units": "mm"},
    "air_density": {"units": "kg m^-3"},
    "air_isentropic_density": {"units": "kg m^-2 K^-1"},
    "air_pressure_on_interface_levels": {"units": "Pa"},
    "air_temperature": {"units": "K"},
    "exner_function_on_interface_levels": {"units": "J K^-1 kg^-1"},
    "height_on_interface_levels": {"units": "m"},
    "mass_fraction_of_cloud_liquid_water_in_air": {"units": "g g^-1"},
    "mass_fraction_of_precipitation_water_in_air": {"units": "g g^-1"},
    "mass_fraction_of_water_vapor_in_air": {"units": "g g^-1"},
    "montgomery_potential": {"units": "m^2 s^-2"},
    "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    "x_velocity_at_u_locations": {"units": "m s^-1"},
    "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    "y_velocity_at_v_locations": {"units": "m s^-1"},
}

tlevels1 = range(0, 19)
tlevels2 = range(0, 19)


# ========================================
# code
# ========================================
def get_range(x, y=None, z=None):
    z = z or 1
    return range(x, y, z) if y is not None else range(0, x, z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f1",
        metavar="filename",
        type=str,
        default=filename1,
        help="Path to the first data set.",
        dest="filename1",
    )
    parser.add_argument(
        "-f2",
        metavar="filename",
        type=str,
        default=filename2,
        help="Path to the second data set.",
        dest="filename2",
    )
    parser.add_argument(
        "-t1",
        metavar=("start", "stop", "step"),
        type=int,
        nargs=3,
        default=(tlevels1.start, tlevels1.stop, tlevels1.step),
        help="Time levels to be considered from the first data set.",
        dest="t1",
    )
    parser.add_argument(
        "-t2",
        metavar=("start", "stop", "step"),
        type=int,
        nargs=3,
        default=(tlevels2.start, tlevels2.stop, tlevels2.step),
        help="Time levels to be considered from the second data set.",
        dest="t2",
    )
    parser.add_argument("-v", help="Verbose output.", dest="verbose", action="store_true")

    args = parser.parse_args()
    fname1 = args.filename1
    fname2 = args.filename2
    verbose = args.verbose

    _, _, states1 = taz.load_netcdf_dataset(fname1)
    _, _, states2 = taz.load_netcdf_dataset(fname2)

    t1s = get_range(*args.t1)
    t2s = get_range(*args.t2)

    validated = True

    for t1, t2 in zip(t1s, t2s):
        state1 = states1[t1]
        state2 = states2[t2]

        iteration_validated = True

        for name, props in field_properties.items():
            units = props.get("units", state1[name].attrs.get("units", None))
            assert units is not None

            field1 = state1[name].to_units(units).values
            field2 = state2[name].to_units(units).values

            isclose = np.allclose(field1, field2, equal_nan=True)

            if not isclose:
                diff = np.abs(field1 - field2)
                rdiff = diff / np.abs(field1)
                rdiff[np.isnan(rdiff)] = 0.0

                if iteration_validated:
                    print("Iteration ({:4d}, {:4d}):".format(t1, t2))
                print(
                    "    Maximum absolute and relative error for {}: {:5E} {:5E}.".format(
                        name, diff.max(), rdiff.max()
                    )
                )
                iteration_validated = False
                validated = False

        if not iteration_validated:
            print("")
        elif verbose:
            print("Iteration ({:4d}, {:4d}) validated!".format(t1, t2))

    if validated:
        print("Solutions validated!")
