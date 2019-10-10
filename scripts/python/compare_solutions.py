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


#
# User inputs
#
filename1 = "../../data/burgers_ssus_reference.nc"
filename2 = "../../data/burgers_ssus_gtmc.nc"

field_properties = {"x_velocity": {"units": "m s^-1"}, "y_velocity": {"units": "m s^-1"}}
# field_properties = {
# 	'accumulated_precipitation': {'units': 'mm'},
# 	'air_density': {'units': 'kg m^-3'},
# 	'air_isentropic_density': {'units': 'kg m^-2 K^-1'},
# 	'air_pressure_on_interface_levels': {'units': 'Pa'},
# 	'air_temperature': {'units': 'K'},
# 	'exner_function_on_interface_levels': {'units': 'J K^-1 kg^-1'},
# 	'height_on_interface_levels': {'units': 'm'},
# 	'mass_fraction_of_cloud_liquid_water_in_air': {'units': 'g g^-1'},
# 	'mass_fraction_of_precipitation_water_in_air': {'units': 'g g^-1'},
# 	'mass_fraction_of_water_vapor_in_air': {'units': 'g g^-1'},
# 	'montgomery_potential': {'units': 'm^2 s^-2'},
# 	'x_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
# 	'x_velocity_at_u_locations': {'units': 'm s^-1'},
# 	'y_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
# 	'y_velocity_at_v_locations': {'units': 'm s^-1'},
# }

tlevels1 = range(0, 401)
tlevels2 = range(0, 401)

#
# Code
#
if __name__ == "__main__":
    _, _, states1 = taz.load_netcdf_dataset(filename1)
    _, _, states2 = taz.load_netcdf_dataset(filename2)

    validated = True

    for t1, t2 in zip(tlevels1, tlevels2):
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

    if validated:
          print("Validation successfully completed!")
