# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2012-2019, ETH Zurich
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

prefix = "../../data/pdc_paper/isentropic_diagnostic/"

reference_dataset = {
    "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx2561_ny1_nz60_"
    "dt0_nt28800_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_ssus_gtx86.nc",
    "xslice": slice(960, 1601),
    "yslice": slice(0, 1),
    "zslice": slice(0, 1),
    "tlevel": 12,
}

datasets = (
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx41_ny1_nz60_"
        "dt40_nt450_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(15, 26),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 64,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx81_ny1_nz60_"
        "dt20_nt900_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(30, 51),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 32,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx161_ny1_nz60_"
        "dt10_nt1800_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(60, 101),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 16,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx321_ny1_nz60_"
        "dt5_nt3600_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(120, 201),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 8,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx641_ny1_nz60_"
        "dt2_nt7200_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(240, 401),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 4,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
    {
        "filename": "isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_nx1281_ny1_nz60_"
        "dt1_nt14400_gaussian_L50000_H500_u22_rh95_lh_smooth_turb_sed_sus_gtx86.nc",
        "xslice": slice(480, 801),
        "yslice": slice(0, 1),
        "zslice": slice(0, 1),
        "xsampling": 2,
        "ysampling": 1,
        "zsampling": 1,
        "tlevel": 12,
    },
)

# ========================================
# code
# ========================================
if __name__ == "__main__":
    # get reference solution
    # print(reference_dataset["filename"])
    _, _, states = taz.load_netcdf_dataset(
        prefix + reference_dataset["filename"]
    )
    state_r = states[reference_dataset["tlevel"]]
    raw_field_r = state_r[field_name].to_units(field_units).values
    refsol = raw_field_r[
        reference_dataset["xslice"],
        reference_dataset["yslice"],
        reference_dataset["zslice"],
    ]

    err = np.zeros(len(datasets))

    # scan all datasets
    for i, ds in enumerate(datasets):
        print(ds["filename"])

        _, _, states = taz.load_netcdf_dataset(prefix + ds["filename"])
        state = states[ds["tlevel"]]
        raw_field = state[field_name].to_units(field_units).data
        sol = raw_field[ds["xslice"], ds["yslice"], ds["zslice"]]
        rsol = refsol[
            :: ds["xsampling"], :: ds["ysampling"], :: ds["zsampling"]
        ]

        # err = (
        #     np.sum(np.abs(sol - rsol) ** 2)
        #     / (sol.shape[0] * sol.shape[1] * sol.shape[2])
        # ) ** 0.5
        err[i] = np.linalg.norm(sol - rsol) / np.sqrt(sol.size)
        # err = np.abs(sol - rsol).max()

        print(f"{err[i]:5.5E}")

    dt = np.linspace(len(datasets), 1, len(datasets))

    p = np.polyfit(dt, np.log2(err), deg=1)

    print(f"EOC: {p[0]:5.5f}")
