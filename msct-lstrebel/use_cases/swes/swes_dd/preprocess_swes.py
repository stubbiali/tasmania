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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains
"""
import numpy as np
import math
import argparse

from dd_preprocess import DomainPreprocess


def prepare_partitioning(nx, ny, nz, sx, sy, sz, nparts, only_advection, px=1, py=0, pz=0, path="", prefix=""):
    cdomain = np.array([nx, ny, nz])
    slices = np.array([sx, sy, sz])
    periodic = np.array([px, py, pz])

    ddc = DomainPreprocess(domain=cdomain, periodic=periodic, subdivs_per_dim=slices, path=path, prefix=prefix)

    # Add IC specific stencils:
    if only_advection:
        ddc.add_stencil({"h": [[0], [1], [0], [1], [0], [0]]})
        ddc.add_stencil({"u": [[0], [1], [0], [0], [0], [0]]})
        ddc.add_stencil({"v": [[0], [0], [0], [1], [0], [0]]})
        ddc.add_stencil({"c": [[0], [0], [0], [1], [0], [0]]})
        ddc.add_stencil({"u_midx": [[1], [0], [0], [0], [0], [0]]})
        ddc.add_stencil({"v_midy": [[0], [0], [1], [0], [0], [0]]})
        ddc.add_stencil({"c_midy": [[0], [0], [1], [0], [0], [0]]})
    else:
        pass

    # Once all stencils are added call preprocess and partitioning
    ddc.preprocess()
    ddc.pymetis_partitioning(nparts)


def prepare_initial_condition(m, n, ic, only_advection, planet, dtype, path="", prefix=""):
    """
    Set the initial conditions.
    """

    if planet == 0:  # Earth
        g = 9.80616
        a = 6.37122e6
        omega = 7.292e-5
        scale_height = 8e3
        nu = 5e5
    elif planet == 1:  # Saturn
        g = 10.44
        a = 5.8232e7
        omega = 2. * math.pi / (10.656 * 3600.)
        scale_height = 60e3
        nu = 5e6
    else:
        raise ValueError('Unknown planet {}.'.format(planet))

    halo = [1, 1, 1, 1, 0, 0]
    # 
    # Lat-lon grid
    #
    assert (m > 1) and (n > 1), \
        "Number of grid points along each direction must be greater than one."
    # Discretize longitude
    # m = m
    dphi = 2. * math.pi / m
    phi_1d = np.linspace(-dphi, 2. * math.pi, m + halo[0] + halo[1], dtype=dtype)

    # Discretize latitude
    # n = n
    # dtheta = math.pi / n
    # dtheta = 170.0 / n
    # Latitude with poles
    #theta_1d = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n, dtype=dtype)
    # Latitude between 85 and -85
    theta_1d = np.linspace(-85.0  / 180.0 * math.pi, 85.0 / 180.0 * math.pi, n, dtype=dtype)
    dtheta = theta_1d[1] - theta_1d[0]
    theta_1d = np.insert(theta_1d, 0, theta_1d[0] - dtheta)
    theta_1d = np.append(theta_1d, theta_1d[-1] + dtheta)

    # Build grid
    phi, theta = np.meshgrid(phi_1d, theta_1d, indexing='ij')

    # Compute cos(theta)
    c = np.cos(theta)
    c_midy = np.cos(0.5 * (theta[:, :-1] + theta[:, 1:]))

    # Compute tan(theta)
    tg = np.tan(theta)
    tg_midx = np.tan(0.5 * (theta[:-1, :] + theta[1:, :]))
    tg_midy = np.tan(0.5 * (theta[:, :-1] + theta[:, 1:]))

    # Coriolis term
    f = 2. * omega * np.sin(theta)

    #
    # Flat terrain height
    #
    hs = np.zeros((m + halo[0] + halo[1], n + halo[2] + halo[3]), dtype=dtype)

    # 
    # Cartesian coordinates and increments
    # 
    x = a * np.cos(theta) * phi
    y = a * theta
    y1 = a * np.sin(theta)

    dx = x[1:, :] - x[:-1, :]
    dy = y[:, 1:] - y[:, :-1]
    dy1 = y1[:, 1:] - y1[:, :-1]

    # Compute minimum longitudinal and latitudinal distance between
    # adjacent grid points, needed to compute timestep size through
    # CFL condition
    dx_min = dx[:, 1:-1].min()
    dy_min = dy.min()

    # "Centred" increments
    dxc = np.zeros((m + halo[0] + halo[1], n + halo[2] + halo[3]), dtype=dtype)
    dxc[1:-1, 1:-1] = 0.5 * (dx[:-1, 1:-1] + dx[1:, 1:-1])
    dyc = np.zeros((m + halo[0] + halo[1], n + halo[2] + halo[3]), dtype=dtype)
    dyc[1:-1, 1:-1] = 0.5 * (dy[1:-1, :-1] + dy[1:-1, 1:])
    dy1c = np.zeros((m + halo[0] + halo[1], n + halo[2] + halo[3]), dtype=dtype)
    dy1c[1:-1, 1:-1] = 0.5 * (dy1[1:-1, :-1] + dy1[1:-1, 1:])

    # Compute longitudinal increment used in pole treatment
    #dxp = 2. * a * np.sin(dtheta) / (1. + np.cos(dtheta))

    # Compute map factor at the poles
    #m_north = 2. / (1. + np.sin(theta_1d[-2]))
    #m_south = 2. / (1. - np.sin(theta_1d[1]))

    #
    # Pre-compute coefficients for second-order three
    # points approximations of first-order derivative
    #
    if diffusion:
        # Centred finite difference along longitude
        # Ax, Bx and Cx denote the coefficients associated
        # with the centred, downwind and upwind point, respectively
        Ax = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        Ax[2:-2, 2:-2] = (dx[1:, 1:-1] - dx[:-1, 1:-1]) / (dx[1:, 1:-1] * dx[:-1, 1:-1])
        Ax[1, :], Ax[-2, :] = Ax[-4, :], Ax[3, :]

        Bx = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        Bx[2:-2, 2:-2] = dx[:-1, 1:-1] / (dx[1:, 1:-1] * (dx[1:, 1:-1] + dx[:-1, 1:-1]))
        Bx[1, :], Bx[-2, :] = Bx[-4, :], Bx[3, :]

        Cx = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        Cx[2:-2, 2:-2] = - dx[1:, 1:-1] / (dx[:-1, 1:-1] * (dx[1:, 1:-1] + dx[:-1, 1:-1]))
        Cx[1, :], Cx[-2, :] = Cx[-4, :], Cx[3, :]

        # Centred finite difference along latitude
        # Ay, By and Cy denote the coefficients associated
        # with the centred, downwind and upwind point, respectively
        Ay = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        Ay[2:-2, 2:-2] = (dy[1:-1, 1:] - dy[1:-1, :-1]) / (dy[1:-1, 1:] * dy[1:-1, :-1])

        By = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        By[2:-2, 2:-2] = dy[1:-1, :-1] / (dy[1:-1, 1:] * (dy[1:-1, 1:] + dy[1:-1, :-1]))
        By[2:-2, -2] = 1. / (2. * dy[1:-1, -2])
        By[2:-2, 1] = 1. / (2. * dy[1:-1, 1])

        Cy = np.zeros((m + halo[0] + halo[1] + 2, n + halo[2] + halo[3] + 2), dtype=dtype)
        Cy[2:-2, 2:-2] = - dy[1:-1, 1:] / (dy[1:-1, :-1] * (dy[1:-1, 1:] + dy[1:-1, :-1]))
        Cy[2:-2, -2] = - 1. / (2. * dy[1:-1, -2])
        Cy[2:-2, 1] = - 1. / (2. * dy[1:-1, 1])

    # 
    # First test case taken from Williamson's suite 
    #
    if ic[0] == 0:
        # Extract alpha
        alpha = ic[1]

        # Define coefficients
        u0 = 2. * math.pi * a / (12. * 24. * 3600.)
        h0 = 1000.
        phi_c = 1.5 * math.pi
        theta_c = 0.
        R = a / 3.

        # Compute steering wind
        u = u0 * (np.cos(theta) * np.cos(alpha) +
                  np.sin(theta) * np.cos(phi) * np.sin(alpha))
        u_midx = np.zeros((m + halo[0] + halo[1] - 1, n + halo[2] + halo[3]), dtype=dtype)
        u_midx[:, 1:-1] = u0 * (np.cos(0.5 * (theta[:-1, 1:-1] + theta[1:, 1:-1])) * np.cos(alpha) +
                                np.sin(0.5 * (theta[:-1, 1:-1] + theta[1:, 1:-1])) *
                                np.cos(0.5 * (phi[:-1, 1:-1] + phi[1:, 1:-1])) * np.sin(alpha))

        v = - u0 * np.sin(phi) * np.sin(alpha)
        v_midy = np.zeros((m + halo[0] + halo[1], n + halo[2] + halo[3] - 1), dtype=dtype)
        v_midy[1:-1, :] = - u0 * np.sin(0.5 * (phi[1:-1, :-1] + phi[1:-1, 1:])) * np.sin(alpha)

        # Compute initial fluid height
        r = a * np.arccos(np.sin(theta_c) * np.sin(theta) +
                               np.cos(theta_c) * np.cos(theta) *
                               np.cos(phi - phi_c))
        h = np.where(r < R, 0.5 * h0 * (1. + np.cos(math.pi * r / R)), 0.)


    #
    # Sixth test case taken from Williamson's suite
    #
    if ic[0] == 2:
        # Set constants
        w = 7.848e-6
        K = 7.848e-6
        h0 = 8e3
        R = 4.

        # Compute initial fluid height
        A = 0.5 * w * (2. * omega + w) * (np.cos(theta) ** 2.) + \
            0.25 * (K ** 2.) * (np.cos(theta) ** (2. * R)) * \
            ((R + 1.) * (np.cos(theta) ** 2.) +
             (2. * (R ** 2.) - R - 2.) -
             2. * (R ** 2.) * (np.cos(theta) ** (-2.)))
        B = (2. * (omega + w) * K) / ((R + 1.) * (R + 2.)) * \
            (np.cos(theta) ** R) * \
            (((R ** 2.) + 2. * R + 2.) -
             ((R + 1.) ** 2.) * (np.cos(theta) ** 2.))
        C = 0.25 * (K ** 2.) * (np.cos(theta) ** (2. * R)) * \
            ((R + 1.) * (np.cos(theta) ** 2.) - (R + 2.))

        h = h0 + ((a ** 2.) * A +
                       (a ** 2.) * B * np.cos(R * phi) +
                       (a ** 2.) * C * np.cos(2. * R * phi)) / g

        # Compute initial wind
        u = a * w * np.cos(theta) + \
                 a * K * (np.cos(theta) ** (R - 1.)) * \
                 (R * (np.sin(theta) ** 2.) - (np.cos(theta) ** 2.)) * \
                 np.cos(R * phi)
        v = - a * K * R * (np.cos(theta) ** (R - 1.)) * \
                 np.sin(theta) * np.sin(R * phi)

    # 
    # Set height at the poles 
    # If values at different longitudes are not\ the same, average them out
    h[:, -1] = np.sum(h[1:-2, -1]) / m
    h[:, 0] = np.sum(h[1:-2, 0]) / m

    # #
    # # Stereographic wind components at the poles
    # #
    # if not only_advection:
    #     # Compute stereographic components at North pole
    #     # at each longitude and then make an average
    #     u_north = - u[1:-2, -1] * np.sin(phi_1d[1:-2]) \
    #               - v[1:-2, -1] * np.cos(phi_1d[1:-2])
    #     u_north = np.sum(u_north) / m
    #
    #     v_north = + u[1:-2, -1] * np.cos(phi_1d[1:-2]) \
    #               - v[1:-2, -1] * np.sin(phi_1d[1:-2])
    #     v_north = np.sum(v_north) / m
    #
    #     # Compute stereographic components at South pole
    #     # at each longitude and then make an average
    #     u_south = - u[1:-2, 0] * np.sin(phi_1d[1:-2]) \
    #               + v[1:-2, 0] * np.cos(phi_1d[1:-2])
    #     u_south = np.sum(u_south) / m
    #
    #     v_south = + u[1:-2, 0] * np.cos(phi_1d[1:-2]) \
    #               + v[1:-2, 0] * np.sin(phi_1d[1:-2])
    #     v_south = np.sum(v_south) / m

    dx = dx.reshape((dx.shape[0], dx.shape[1], 1))
    dxc = dxc.reshape((dxc.shape[0], dxc.shape[1], 1))
    dy = dy.reshape((dy.shape[0], dy.shape[1], 1))
    dyc = dyc.reshape((dyc.shape[0], dyc.shape[1], 1))
    dy1 = dy1.reshape((dy1.shape[0], dy1.shape[1], 1))
    dy1c = dy1c.reshape((dy1c.shape[0], dy1c.shape[1], 1))
    c = c.reshape((c.shape[0], c.shape[1], 1))
    c_midy = c_midy.reshape((c_midy.shape[0], c_midy.shape[1], 1))

    f = f.reshape((dy.shape[0], f.shape[1], 1))

    u = u.reshape((u.shape[0], u.shape[1], 1))
    v = v.reshape((v.shape[0], v.shape[1], 1))
    h = h.reshape((h.shape[0], h.shape[1], 1))

    hs = hs.reshape((hs.shape[0], hs.shape[1], 1))
    tg = tg.reshape((tg.shape[0], tg.shape[1], 1))
    tg_midx = tg_midx.reshape((tg_midx.shape[0], tg_midx.shape[1], 1))
    tg_midy = tg_midy.reshape((tg_midy.shape[0], tg_midy.shape[1], 1))

    if ic[0] == 0:
        u_midx = u_midx.reshape((u_midx.shape[0], u_midx.shape[1], 1))
        v_midy = v_midy.reshape((v_midy.shape[0], v_midy.shape[1], 1))

    if diffusion:
        Ax = Ax.reshape((Ax.shape[0], Ax.shape[1], 1))
        Bx = Bx.reshape((Bx.shape[0], Bx.shape[1], 1))
        Cx = Cx.reshape((Cx.shape[0], Cx.shape[1], 1))
        Ay = Ay.reshape((Ay.shape[0], Ay.shape[1], 1))
        By = By.reshape((By.shape[0], By.shape[1], 1))
        Cy = Cy.reshape((Cy.shape[0], Cy.shape[1], 1))

    # print(dx.shape, dxc.shape, dy.shape, dy1.shape, dy1c.shape, c.shape, c_midy.shape,
    #       u.shape, v.shape, h.shape, u_midx.shape, v_midy.shape)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_phi.npy", phi)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_theta.npy", theta)

    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dx.npy", dx)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dxc.npy", dxc)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dy.npy", dy)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dy1.npy", dy1)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dy1c.npy", dy1c)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_c.npy", c)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_c_midy.npy", c_midy)

    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_u.npy", u)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_v.npy", v)
    np.save(path + prefix + "swes_ic" + str(ic[0]) + "_h.npy", h)

    if ic[0] == 0:
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_u_midx.npy", u_midx)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_v_midy.npy", v_midy)

    if diffusion:
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_ax.npy", Ax)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_bx.npy", Bx)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_cx.npy", Cx)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_ay.npy", Ay)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_by.npy", By)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_cy.npy", Cy)

    if not only_advection:
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_dyc.npy", dyc)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_f.npy", f)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_hs.npy", hs)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_tg.npy", tg)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_tg_midx.npy", tg_midx)
        np.save(path + prefix + "swes_ic" + str(ic[0]) + "_tg_midy.npy", tg_midy)

        # np.save(path + prefix + "swes_ic" + str(ic[0]) + "_u_north.npy", u_north)
        # np.save(path + prefix + "swes_ic" + str(ic[0]) + "_u_south.npy", u_south)
        # np.save(path + prefix + "swes_ic" + str(ic[0]) + "_v_north.npy", v_north)
        # np.save(path + prefix + "swes_ic" + str(ic[0]) + "_v_south.npy", v_south)



if __name__ == "__main__":
    # Suggested values for $\alpha$ for first and second
    # test cases from Williamson's suite:
    # * 0
    # * 0.05
    # * pi/2 - 0.05
    # * pi/2
    ic = (2, 0) #math.pi / 2)

    m = 180
    n = 90
    nz = 1
    sx = 2
    sy = 2
    sz = 1
    nparts = 2

    if ic[0] == 0:
        only_advection = True
    else:
        only_advection = False

    diffusion = True

    path = ""
    prefix = ""
    # ny=n-2
    prepare_partitioning(nx=m, ny=n, nz=nz, sx=sx, sy=sy, sz=sz, nparts=nparts, only_advection=only_advection,
                         path=path, prefix=prefix)

    prepare_initial_condition(m=m, n=n, ic=ic, only_advection=only_advection, planet=0, dtype=np.float64,
                              path=path, prefix=prefix)
