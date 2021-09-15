# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import tasmania as taz


def main():
    filename1 = (
        "../../data/pdc_paper/isentropic_prognostic/isentropic_moist_rk3ws_si_"
        "fifth_order_upwind_nx161_ny1_nz60_dt10_nt1800_gaussian_L50000_H500_"
        "u22_rh95_lh_smooth_turb_sed_fc_gtx86.nc"
    )
    filename2 = (
        "../../data/pdc_paper/isentropic_prognostic/20210826/isentropic_moist_rk3ws_si_"
        "fifth_order_upwind_nx161_ny1_nz90_dt10_nt1800_gaussian_L50000_H500_"
        "u22_rh95_lh_smooth_turb_sed_fc_gt4py:gtmc_oop.nc"
    )
    t = 18

    _, _, states1 = taz.load_netcdf_dataset(filename1)
    prec1 = states1[t]["precipitation"].data[:, 0, 0]
    _, _, states2 = taz.load_netcdf_dataset(filename2)
    prec2 = states2[t]["precipitation"].data[:, 0, 0]

    x = np.arange(0, prec1.shape[0])

    fig = plt.figure(figsize=(6, 6))
    plt.plot(x, prec1, "b")
    plt.plot(x, prec2, "r")
    plt.show()


if __name__ == "__main__":
    main()
