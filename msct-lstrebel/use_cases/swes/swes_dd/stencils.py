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
"""
This module contains the definitions of some GT4Py stencils for solving the Shallow Water Equation on a Sphere.
"""
import gridtools as gt

def definitions_advection(dt, dx, dxc, dy1, dy1c, c, c_midy,
                          u, u_midx, v, v_midy, in_h):
    """
    Definitions function for the GT4Py stencil integrating
    pure advection equations.
    """
    #
    # Indices
    #
    i = gt.Index()
    j = gt.Index()

    #
    # Temporary and output arrays
    #
    h_midx = gt.Equation()
    h_midy = gt.Equation()
    out_h = gt.Equation()

    #
    # Computations
    #
    h_midx[i, j] = (0.5 * (in_h[i, j] + in_h[i + 1, j]) -
                    0.5 * dt / dx[i, j] *
                    (in_h[i + 1, j] * u[i + 1, j] - in_h[i, j] * u[i, j]))
    h_midy[i, j] = (0.5 * (in_h[i, j] + in_h[i, j + 1]) -
                    0.5 * dt / dy1[i, j] *
                    (in_h[i, j + 1] * v[i, j + 1] * c[i, j + 1] -
                    in_h[i, j] * v[i, j] * c[i, j]))
    out_h[i, j] = (in_h[i, j] -
                   dt / dxc[i, j] * (h_midx[i, j] * u_midx[i, j] - h_midx[i - 1, j] * u_midx[i - 1, j]) -
                   dt / dy1c[i, j] *
                   (h_midy[i, j] * v_midy[i, j] * c_midy[i, j] - h_midy[i, j - 1] * v_midy[i, j - 1] * c_midy[i, j - 1])
                   )

    return out_h


def definitions_lax_wendroff(dt, dx, dxc, dy, dyc, dy1, dy1c,
                             c, c_midy, f, a, g, hs, tg, tg_midx, tg_midy,
                             in_h, in_u, in_v):
    """
    Definitions function for the GT4Py stencil implementing the
    Lax-Wendroff scheme for the shallow water equations defined
    on a sphere.
    """
    #
    # Indices
    #
    i = gt.Index()
    j = gt.Index()

    #
    # Temporary and output arrays
    #
    v1 = gt.Equation()
    hu = gt.Equation()
    hv = gt.Equation()
    hv1 = gt.Equation()

    h_midx = gt.Equation()
    Ux = gt.Equation()
    hu_midx = gt.Equation()
    Vx = gt.Equation()
    hv_midx = gt.Equation()

    h_midy = gt.Equation()
    Uy = gt.Equation()
    hu_midy = gt.Equation()
    Vy1 = gt.Equation()
    Vy2 = gt.Equation()
    hv_midy = gt.Equation()

    out_h = gt.Equation()

    Ux_mid = gt.Equation()
    Uy_mid = gt.Equation()
    out_hu = gt.Equation()
    out_u = gt.Equation()

    Vx_mid = gt.Equation()
    Vy1_mid = gt.Equation()
    Vy2_mid = gt.Equation()
    out_hv = gt.Equation()
    out_v = gt.Equation()

    #
    # Compute
    #
    v1[i, j] = in_v[i, j] * c[i, j]
    hu[i, j] = in_h[i, j] * in_u[i, j]
    hv[i, j] = in_h[i, j] * in_v[i, j]
    hv1[i, j] = hv[i, j] * c[i, j]

    # Longitudinal mid-values for h
    h_midx[i, j] = 0.5 * (in_h[i, j] + in_h[i + 1, j]) - \
                   0.5 * dt / dx[i, j] * (hu[i + 1, j] - hu[i, j])

    # Longitudinal mid-values for hu
    Ux[i, j] = hu[i, j] * in_u[i, j] + 0.5 * g * in_h[i, j] * in_h[i, j]
    hu_midx[i, j] = 0.5 * (hu[i, j] + hu[i + 1, j]) - \
                    0.5 * dt / dx[i, j] * (Ux[i + 1, j] - Ux[i, j]) + \
                    0.5 * dt * \
                    (0.5 * (f[i, j] + f[i + 1, j]) +
                     0.5 * (in_u[i, j] + in_u[i + 1, j]) * tg_midx[i, j] / a) * \
                    0.5 * (hv[i, j] + hv[i + 1, j])

    # Longitudinal mid-values for hv
    Vx[i, j] = hu[i, j] * in_v[i, j]
    hv_midx[i, j] = 0.5 * (hv[i, j] + hv[i + 1, j]) - \
                    0.5 * dt / dx[i, j] * (Vx[i + 1, j] - Vx[i, j]) - \
                    0.5 * dt * \
                    (0.5 * (f[i, j] + f[i + 1, j]) +
                     0.5 * (in_u[i, j] + in_u[i + 1, j]) * tg_midx[i, j] / a) * \
                    0.5 * (hu[i, j] + hu[i + 1, j])

    # Latitudinal mid-values for h
    h_midy[i, j] = 0.5 * (in_h[i, j] + in_h[i, j + 1]) - \
                   0.5 * dt / dy1[i, j] * (hv1[i, j + 1] - hv1[i, j])

    # Latitudinal mid-values for hu
    Uy[i, j] = hu[i, j] * v1[i, j]
    hu_midy[i, j] = 0.5 * (hu[i, j] + hu[i, j + 1]) - \
                    0.5 * dt / dy1[i, j] * (Uy[i, j + 1] - Uy[i, j]) + \
                    0.5 * dt * \
                    (0.5 * (f[i, j] + f[i, j + 1]) +
                     0.5 * (in_u[i, j] + in_u[i, j + 1]) * tg_midy[i, j] / a) * \
                    0.5 * (hv[i, j] + hv[i, j + 1])

    # Latitudinal mid-values for hv
    Vy1[i, j] = hv[i, j] * v1[i, j]
    Vy2[i, j] = 0.5 * g * in_h[i, j] * in_h[i, j]
    hv_midy[i, j] = 0.5 * (hv[i, j] + hv[i, j + 1]) - \
                    0.5 * dt / dy1[i, j] * (Vy1[i, j + 1] - Vy1[i, j]) - \
                    0.5 * dt / dy[i, j] * (Vy2[i, j + 1] - Vy2[i, j]) - \
                    0.5 * dt * \
                    (0.5 * (f[i, j] + f[i, j + 1]) +
                     0.5 * (in_u[i, j] + in_u[i, j + 1]) * tg_midy[i, j] / a) * \
                    0.5 * (hu[i, j] + hu[i, j + 1])

    # Update h
    out_h[i, j] = in_h[i, j] - \
                  dt / dxc[i, j] * (hu_midx[i, j] - hu_midx[i - 1, j]) - \
                  dt / dy1c[i, j] * (hv_midy[i, j] * c_midy[i, j] -
                                     hv_midy[i, j - 1] * c_midy[i, j - 1])

    # Update hu
    Ux_mid[i, j] = (h_midx[i, j] > 0.) * hu_midx[i, j] * hu_midx[i, j] / h_midx[i, j] + \
                   0.5 * g * h_midx[i, j] * h_midx[i, j]
    Uy_mid[i, j + 0.5] = (h_midy[i, j] > 0.) * hv_midy[i, j] * c_midy[i, j] * \
                         hu_midy[i, j] / h_midy[i, j]
    out_hu[i, j] = hu[i, j] - \
                   dt / dxc[i, j] * (Ux_mid[i, j] - Ux_mid[i - 1, j]) - \
                   dt / dy1c[i, j] * (Uy_mid[i, j] - Uy_mid[i, j - 1]) + \
                   dt * (f[i, j] +
                         0.25 * (hu_midx[i - 1, j] / h_midx[i - 1, j] +
                                 hu_midx[i, j] / h_midx[i, j] +
                                 hu_midy[i, j - 1] / h_midy[i, j - 1] +
                                 hu_midy[i, j] / h_midy[i, j]) *
                         tg[i, j] / a) * \
                   0.25 * (hv_midx[i - 1, j] + hv_midx[i, j] +
                           hv_midy[i, j - 1] + hv_midy[i, j]) - \
                   dt * g * \
                   0.25 * (h_midx[i - 1, j] + h_midx[i, j] +
                           h_midy[i, j - 1] + h_midy[i, j]) * \
                   (hs[i + 1, j] - hs[i - 1, j]) / (dx[i - 1, j] + dx[i, j])

    # Update hv
    Vx_mid[i, j] = (h_midx[i, j] > 0.) * hv_midx[i, j] * hu_midx[i, j] / h_midx[i, j]
    Vy1_mid[i, j] = (h_midy[i, j] > 0.) * hv_midy[i, j] * hv_midy[i, j] / \
                    h_midy[i, j] * c_midy[i, j]
    Vy2_mid[i, j] = 0.5 * g * h_midy[i, j] * h_midy[i, j]
    out_hv[i, j] = hv[i, j] - \
                   dt / dxc[i, j] * (Vx_mid[i, j] - Vx_mid[i - 1, j]) - \
                   dt / dy1c[i, j] * (Vy1_mid[i, j] - Vy1_mid[i, j - 1]) - \
                   dt / dyc[i, j] * (Vy2_mid[i, j] - Vy2_mid[i, j - 1]) - \
                   dt * (f[i, j] +
                         0.25 * (hu_midx[i - 1, j] / h_midx[i - 1, j] +
                                 hu_midx[i, j] / h_midx[i, j] +
                                 hu_midy[i, j - 1] / h_midy[i, j - 1] +
                                 hu_midy[i, j] / h_midy[i, j]) *
                         tg[i, j] / a) * \
                   0.25 * (hu_midx[i - 1, j] + hu_midx[i, j] +
                           hu_midy[i, j - 1] + hu_midy[i, j]) - \
                   dt * g * \
                   0.25 * (h_midx[i - 1, j] + h_midx[i, j] +
                           h_midy[i, j - 1] + h_midy[i, j]) * \
                   (hs[i, j + 1] - hs[i, j - 1]) / (dy1[i, j - 1] + dy1[i, j])

    # Come back to original variables
    out_u[i, j] = out_hu[i, j] / out_h[i, j]
    out_v[i, j] = out_hv[i, j] / out_h[i, j]

    return out_h, out_u, out_v


def definitions_diffusion(dt, Ax, Ay, Bx, By, Cx, Cy, nu, in_q, tmp_q):
    """
    Definitions function for the GT4Py stencil applying numerical diffusion.
    """
    #
    # Indices
    #
    i = gt.Index()
    j = gt.Index()

    #
    # Temporary and output arrays
    #
    qxx = gt.Equation()
    qyy = gt.Equation()
    out_q = gt.Equation()

    #
    # Compute
    #
    qxx[i, j] = Ax[i, j] * (Ax[i, j] * in_q[i, j] +
                            Bx[i, j] * in_q[i + 1, j] +
                            Cx[i, j] * in_q[i - 1, j]) + \
                Bx[i, j] * (Ax[i + 1, j] * in_q[i + 1, j] +
                            Bx[i + 1, j] * in_q[i + 2, j] +
                            Cx[i + 1, j] * in_q[i, j]) + \
                Cx[i, j] * (Ax[i - 1, j] * in_q[i - 1, j] +
                            Bx[i - 1, j] * in_q[i, j] +
                            Cx[i - 1, j] * in_q[i - 2, j])

    qyy[i, j] = Ay[i, j] * (Ay[i, j] * in_q[i, j] +
                            By[i, j] * in_q[i, j + 1] +
                            Cy[i, j] * in_q[i, j - 1]) + \
                By[i, j] * (Ay[i, j + 1] * in_q[i, j + 1] +
                            By[i, j + 1] * in_q[i, j + 2] +
                            Cy[i, j + 1] * in_q[i, j]) + \
                Cy[i, j] * (Ay[i, j - 1] * in_q[i, j - 1] +
                            By[i, j - 1] * in_q[i, j] +
                            Cy[i, j - 1] * in_q[i, j - 2])

    out_q[i, j] = tmp_q[i, j] + dt * nu * (qxx[i, j] + qyy[i, j])

    return out_q
