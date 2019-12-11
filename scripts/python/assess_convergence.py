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

# ========================================
# user inputs
# ========================================
field_name = "precipitation"
field_units = "mm hr^-1"

prefix = "../../data/diagnostic-saturation-290/"

reference_dataset = {
    "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_nx321_ny321_nz60_"
    "dt5_nt720_gaussian_L50000_H1000_u15_rh90_turb_sed_evap_fc_gtx86.nc",
    "xslice": slice(80, 241),
    "yslice": slice(80, 241),
    "zslice": slice(0, 1),
    "tlevel": 25,
}

datasets = (
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx41_ny41_nz60_"
        "dt40_nt90_gaussian_L50000_H1000_u15_rh90_turb_sed_evap_sus_gtx86.nc",
        "xslice": slice(10, 31),
        "yslice": slice(10, 31),
        "zslice": slice(0, 1),
        "xsampling": 8,
        "ysampling": 8,
        "zsampling": 1,
        "tlevel": 25,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx81_ny81_nz60_"
        "dt20_nt180_gaussian_L50000_H1000_u15_rh90_turb_sed_evap_sus_gtx86.nc",
        "xslice": slice(20, 61),
        "yslice": slice(20, 61),
        "zslice": slice(0, 1),
        "xsampling": 4,
        "ysampling": 4,
        "zsampling": 1,
        "tlevel": 25,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx161_ny161_nz60_"
        "dt10_nt360_gaussian_L50000_H1000_u15_rh90_turb_sed_evap_sus_gtx86.nc",
        "xslice": slice(40, 121),
        "yslice": slice(40, 121),
        "zslice": slice(0, 1),
        "xsampling": 2,
        "ysampling": 2,
        "zsampling": 1,
        "tlevel": 25,
    },
)

# ========================================
# code
# ========================================
if __name__ == "__main__":
    # get reference solution
    print(reference_dataset["filename"])
    _, _, states = taz.load_netcdf_dataset(prefix + reference_dataset["filename"])
    state_r = states[reference_dataset["tlevel"]]
    raw_field_r = state_r[field_name].to_units(field_units).values
    refsol = raw_field_r[
        reference_dataset["xslice"],
        reference_dataset["yslice"],
        reference_dataset["zslice"],
    ]

    # scan all datasets
    for ds in datasets:
        print(ds["filename"])

        _, _, states = taz.load_netcdf_dataset(prefix + ds["filename"])
        state = states[ds["tlevel"]]
        raw_field = state[field_name].to_units(field_units).values
        sol = raw_field[ds["xslice"], ds["yslice"], ds["zslice"]]
        rsol = refsol[:: ds["xsampling"], :: ds["ysampling"], :: ds["zsampling"]]

        err = (
            np.sum(np.abs(sol - rsol) ** 2) / (sol.shape[0] * sol.shape[1] * sol.shape[2])
        ) ** 0.5
        print("{:5.5E}".format(err))
