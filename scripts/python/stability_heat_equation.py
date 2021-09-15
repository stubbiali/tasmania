# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def gf_rk2(dx, r):
    return (
        1
        + r * (2 * np.cos(dx) - 2)
        + 0.5 * r ** 2 * (2 * np.cos(2 * dx) - 8 * np.cos(dx) + 6)
    )


def gf_rk3ws(dx, r):
    return (
        1
        + r * (2 * np.cos(dx) - 2)
        + 0.5 * r ** 2 * (2 * np.cos(2 * dx) - 8 * np.cos(dx) + 6)
        + 1.0
        / 6.0
        * r ** 3
        * (2 * np.cos(3 * dx) - 12 * np.cos(2 * dx) + 30 * np.cos(dx) - 20)
    )


def main(gf):
    dxs = np.linspace(0, 2 * np.pi, 1001)
    r = np.linspace(0, 1, 40001)

    dxmax = 0
    rmax = r.max()

    fig = plt.figure(figsize=(6, 6))

    for dx in dxs:
        # g = (
        #     1
        #     + 2 * r * (np.cos(dx) - 1)
        #     + 1 * r ** 2 * (np.cos(2 * dx) - 4 * np.cos(dx) + 3)
        # )
        g = gf(dx, r)
        if np.any(np.abs(g) > 1):
            rmax_n = r[np.where(np.abs(g) > 1)].min()
            if rmax_n < rmax:
                dxmax = dx
                rmax = rmax_n

        plt.plot(r, gf(dx, r), "b-")

    print(f"rmax = {rmax}")

    plt.show()


if __name__ == "__main__":
    main(gf_rk2)
