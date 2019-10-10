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
This module contains:
    Centered(IsentropicNonconservativeHorizontalFlux)
"""
import gridtools as gt
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicNonconservativeHorizontalFlux,
)


class Centered(IsentropicNonconservativeHorizontalFlux):
    """
    A centered scheme to compute the horizontal
    numerical fluxes for the prognostic model variables.
    The nonconservative form of the governing equations,
    expressed using isentropic coordinates, is used.

    Attributes
    ----------
    nb : int
        Number of boundary layers.
    order : int
        Order of accuracy.
    """

    def __init__(self, grid, moist_on):
        super().__init__(grid, moist_on)
        self.nb = 1
        self.order = 2

    def __call__(self, i, j, k, dt, s, u, v, mtg, qv=None, qc=None, qr=None):
        # Compute the fluxes for the isentropic density and
        # the velocity components
        flux_s_x = self._get_centered_flux_x_s(i, j, k, u, s)
        flux_s_y = self._get_centered_flux_y_s(i, j, k, v, s)
        flux_u_x = self._get_centered_flux_x_u(i, j, k, u)
        flux_u_y = self._get_centered_flux_y_unstg(i, j, k, u)
        flux_v_x = self._get_centered_flux_x_unstg(i, j, k, v)
        flux_v_y = self._get_centered_flux_y_v(i, j, k, v)

        # Initialize the return list
        return_list = [flux_s_x, flux_s_y, flux_u_x, flux_u_y, flux_v_x, flux_v_y]

        if self._moist_on:
            # Compute the fluxes for the water constituents
            flux_qv_x = self._get_centered_flux_x_unstg(i, j, k, qv)
            flux_qv_y = self._get_centered_flux_y_unstg(i, j, k, qv)
            flux_qc_x = self._get_centered_flux_x_unstg(i, j, k, qc)
            flux_qc_y = self._get_centered_flux_y_unstg(i, j, k, qc)
            flux_qr_x = self._get_centered_flux_x_unstg(i, j, k, qr)
            flux_qr_y = self._get_centered_flux_y_unstg(i, j, k, qr)

            # Update the return list
            return_list += [
                flux_qv_x,
                flux_qv_y,
                flux_qc_x,
                flux_qc_y,
                flux_qr_x,
                flux_qr_y,
            ]

        return return_list

    @staticmethod
    def _get_centered_flux_x_s(i, j, k, u, s):
        flux_s_x = gt.Equation()
        flux_s_x[i, j, k] = (
            0.25 * (u[i, j, k] + u[i + 1, j, k]) * s[i, j, k]
            + 0.25 * (u[i - 1, j, k] + u[i, j, k]) * s[i - 1, j, k]
        )
        return flux_s_x

    @staticmethod
    def _get_centered_flux_x_u(i, j, k, u):
        flux_u_x = gt.Equation()
        flux_u_x[i, j, k] = 0.5 * (u[i, j, k] + u[i + 1, j, k])
        return flux_u_x

    @staticmethod
    def _get_centered_flux_x_unstg(i, j, k, phi):
        phi_name = phi.get_name()
        flux_name = "flux_" + phi_name + "_x"
        flux = gt.Equation(name=flux_name)

        flux[i, j, k] = 0.5 * (phi[i - 1, j, k] + phi[i, j, k])

        return flux

    @staticmethod
    def _get_centered_flux_y_s(i, j, k, v, s):
        flux_s_y = gt.Equation()
        flux_s_y[i, j, k] = (
            0.25 * (v[i, j, k] + v[i, j + 1, k]) * s[i, j, k]
            + 0.25 * (v[i, j - 1, k] + v[i, j, k]) * s[i, j - 1, k]
        )
        return flux_s_y

    @staticmethod
    def _get_centered_flux_y_v(i, j, k, v):
        flux_v_y = gt.Equation()
        flux_v_y[i, j, k] = 0.5 * (v[i, j, k] + v[i, j + 1, k])
        return flux_v_y

    @staticmethod
    def _get_centered_flux_y_unstg(i, j, k, phi):
        phi_name = phi.get_name()
        flux_name = "flux_" + phi_name + "_y"
        flux = gt.Equation(name=flux_name)

        flux[i, j, k] = 0.5 * (phi[i, j - 1, k] + phi[i, j, k])

        return flux
