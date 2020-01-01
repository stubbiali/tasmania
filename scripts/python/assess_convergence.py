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
field_name = "x_momentum_isentropic"
field_units = "kg m^-1 K^-1 s^-1"

prefix = "../../data/"

reference_dataset = {
    "filename": "isentropic_dry_rk3ws_si_fifth_order_upwind_nx801_nz300_"
    "dt1_nt60000_gaussian_L10000_H1_u10_T250_turb_fc_gtx86.nc",
    "xslice": slice(240, 561),
    "yslice": slice(0, 1),
    "zslice": slice(200, 300),
    "tlevel": -1,
}

datasets = (
    {
        "filename": "isentropic_dry_rk3ws_si_fifth_order_upwind_nx51_nz300_"
        "dt16_nt3750_gaussian_L10000_H1_u10_T250_turb_fc_gtx86.nc",
        "xslice": slice(15, 36),
        "yslice": slice(0, 1),
        "zslice": slice(200, 300),
        "xsampling": 16,
        "ysampling": 16,
        "zsampling": 1,
        "tlevel": -1,
    },
    {
        "filename": "isentropic_dry_rk3ws_si_fifth_order_upwind_nx101_nz300_"
        "dt8_nt7500_gaussian_L10000_H1_u10_T250_turb_fc_gtx86.nc",
        "xslice": slice(30, 71),
        "yslice": slice(0, 1),
        "zslice": slice(200, 300),
        "xsampling": 8,
        "ysampling": 8,
        "zsampling": 1,
        "tlevel": -1,
    },
    {
        "filename": "isentropic_dry_rk3ws_si_fifth_order_upwind_nx201_nz300_"
        "dt4_nt15000_gaussian_L10000_H1_u10_T250_turb_fc_gtx86.nc",
        "xslice": slice(60, 141),
        "yslice": slice(0, 1),
        "zslice": slice(200, 300),
        "xsampling": 4,
        "ysampling": 4,
        "zsampling": 1,
        "tlevel": -1,
    },
    {
        "filename": "isentropic_dry_rk3ws_si_fifth_order_upwind_nx401_nz300_"
        "dt2_nt30000_gaussian_L10000_H1_u10_T250_turb_fc_gtx86.nc",
        "xslice": slice(120, 281),
        "yslice": slice(0, 1),
        "zslice": slice(200, 300),
        "xsampling": 2,
        "ysampling": 2,
        "zsampling": 1,
        "tlevel": -1,
    },
)

# ========================================
# code
# ========================================
if __name__ == "__main__":
    # get reference solution
    # print(reference_dataset["filename"])
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
