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
from collections.abc import Sequence
import numpy as np
from sympl import DataArray

from tasmania.python.domain.domain import Domain
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.utils.io import load_netcdf_dataset

from drivers.isentropic_diagnostic.utils import print_info


def main(filename, length, store_names):
    domain, grid_type, states = load_netcdf_dataset(filename)

    assert isinstance(domain, Domain)
    assert isinstance(grid_type, str)
    assert grid_type == "physical"
    assert isinstance(states, Sequence)
    assert len(states) == length

    for state in states:
        assert isinstance(state, dict)
        assert "time" in state

        for name in state:
            if name != "time":
                assert name in store_names
                assert isinstance(state[name], DataArray)
                assert isinstance(state[name].data, np.ndarray)

        s = to_numpy(state["air_isentropic_density"].data)
        su = to_numpy(state["x_momentum_isentropic"].data)
        u = su / s
        print(f"umax = {u.max():10.8f}, umin = {u.min():10.8f}")


if __name__ == "__main__":
    filename = "../../data/test.nc"
    length = 19
    store_names = (
        "accumulated_precipitation",
        "air_density",
        "air_isentropic_density",
        "air_pressure_on_interface_levels",
        "air_temperature",
        "exner_function_on_interface_levels",
        "height_on_interface_levels",
        "mass_fraction_of_water_vapor_in_air",
        "mass_fraction_of_cloud_liquid_water_in_air",
        "mass_fraction_of_precipitation_water_in_air",
        "montgomery_potential",
        "precipitation",
        "x_momentum_isentropic",
        "x_velocity_at_u_locations",
        "y_momentum_isentropic",
        "y_velocity_at_v_locations",
    )
    main(filename, length, store_names)
