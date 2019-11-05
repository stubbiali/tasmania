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
import abc


from gridtools import gtscript


class BurgersAdvection(abc.ABC):
    """ A discretizer for the 2-D Burgers advection flux. """

    extent = None

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(dx, dy, u, v):
        pass

    @staticmethod
    def factory(flux_scheme):
        if flux_scheme == "first_order":
            return _FirstOrder()
        elif flux_scheme == "second_order":
            return _SecondOrder()
        elif flux_scheme == "third_order":
            return _ThirdOrder()
        elif flux_scheme == "fourth_order":
            return _FourthOrder()
        elif flux_scheme == "fifth_order":
            return _FifthOrder()
        elif flux_scheme == "sixth_order":
            return _SixthOrder()
        else:
            raise RuntimeError()


class _FirstOrder(BurgersAdvection):
    extent = 1

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        abs_u = u[0, 0, 0] * (u[0, 0, 0] > 0.0) - u[0, 0, 0] * (u[0, 0, 0] < 0)
        abs_v = v[0, 0, 0] * (v[0, 0, 0] > 0.0) - v[0, 0, 0] * (v[0, 0, 0] < 0)

        adv_u_x = u[0, 0, 0] / (2.0 * dx) * (u[+1, 0, 0] - u[-1, 0, 0]) - abs_u[
            0, 0, 0
        ] / (2.0 * dx) * (u[+1, 0, 0] - 2.0 * u[0, 0, 0] + u[-1, 0, 0])
        adv_u_y = v[0, 0, 0] / (2.0 * dy) * (u[0, +1, 0] - u[0, -1, 0]) - abs_v[
            0, 0, 0
        ] / (2.0 * dy) * (u[0, +1, 0] - 2.0 * u[0, 0, 0] + u[0, -1, 0])
        adv_v_x = u[0, 0, 0] / (2.0 * dx) * (v[+1, 0, 0] - v[-1, 0, 0]) - abs_u[
            0, 0, 0
        ] / (2.0 * dx) * (v[+1, 0, 0] - 2.0 * v[0, 0, 0] + v[-1, 0, 0])
        adv_v_y = v[0, 0, 0] / (2.0 * dy) * (v[0, +1, 0] - v[0, -1, 0]) - abs_v[
            0, 0, 0
        ] / (2.0 * dy) * (v[0, +1, 0] - 2.0 * v[0, 0, 0] + v[0, -1, 0])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _SecondOrder(BurgersAdvection):
    extent = 1

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        adv_u_x = u[0, 0, 0] / (2.0 * dx) * (u[+1, 0, 0] - u[-1, 0, 0])
        adv_u_y = v[0, 0, 0] / (2.0 * dy) * (u[0, +1, 0] - u[0, -1, 0])
        adv_v_x = u[0, 0, 0] / (2.0 * dx) * (v[+1, 0, 0] - v[-1, 0, 0])
        adv_v_y = v[0, 0, 0] / (2.0 * dy) * (v[0, +1, 0] - v[0, -1, 0])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _ThirdOrder(BurgersAdvection):
    extent = 2

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        abs_u = u[0, 0, 0] * (u[0, 0, 0] > 0.0) - u[0, 0, 0] * (u[0, 0, 0] < 0)
        abs_v = v[0, 0, 0] * (v[0, 0, 0] > 0.0) - v[0, 0, 0] * (v[0, 0, 0] < 0)

        adv_u_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (u[+1, 0, 0] - u[-1, 0, 0]) - (u[+2, 0, 0] - u[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            u[+2, 0, 0] + u[-2, 0, 0]
            - 4.0 * (u[+1, 0, 0] + u[-1, 0, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_u_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (u[0, +1, 0] - u[0, -1, 0]) - (u[0, +2, 0] - u[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            u[0, +2, 0] + u[0, -2, 0]
            - 4.0 * (u[0, +1, 0] + u[0, -1, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_v_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (v[+1, 0, 0] - v[-1, 0, 0]) - (v[+2, 0, 0] - v[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            v[+2, 0, 0] + v[-2, 0, 0]
            - 4.0 * (v[+1, 0, 0] + v[-1, 0, 0])
            + 6.0 * v[0, 0, 0]
        )
        adv_v_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (v[0, +1, 0] - v[0, -1, 0]) - (v[0, +2, 0] - v[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            v[0, +2, 0] + v[0, -2, 0]
            - 4.0 * (v[0, +1, 0] + v[0, -1, 0])
            + 6.0 * v[0, 0, 0]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _FourthOrder(BurgersAdvection):
    extent = 2

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        adv_u_x = (
            u[0, 0, 0]
            / (12.0 * dx)
            * (8.0 * (u[+1, 0, 0] - u[-1, 0, 0]) - (u[+2, 0, 0] - u[-2, 0, 0]))
        )
        adv_u_y = (
            v[0, 0, 0]
            / (12.0 * dy)
            * (8.0 * (u[0, +1, 0] - u[0, -1, 0]) - (u[0, +2, 0] - u[0, -2, 0]))
        )
        adv_v_x = (
            u[0, 0, 0]
            / (12.0 * dx)
            * (8.0 * (v[+1, 0, 0] - v[-1, 0, 0]) - (v[+2, 0, 0] - v[-2, 0, 0]))
        )
        adv_v_y = (
            v[0, 0, 0]
            / (12.0 * dy)
            * (8.0 * (v[0, +1, 0] - v[0, -1, 0]) - (v[0, +2, 0] - v[0, -2, 0]))
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _FifthOrder(BurgersAdvection):
    extent = 3

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        abs_u = u[0, 0, 0] * (u[0, 0, 0] >= 0.0) - u[0, 0, 0] * (u[0, 0, 0] < 0)
        abs_v = v[0, 0, 0] * (v[0, 0, 0] >= 0.0) - v[0, 0, 0] * (v[0, 0, 0] < 0)

        adv_u_x = u[0, 0, 0] / (60.0 * dx) * (
            +45.0 * (u[+1, 0, 0] - u[-1, 0, 0])
            - 9.0 * (u[+2, 0, 0] - u[-2, 0, 0])
            + (u[+3, 0, 0] - u[-3, 0, 0])
        ) - abs_u[0, 0, 0] / (60.0 * dx) * (
            +(u[+3, 0, 0] + u[-3, 0, 0])
            - 6.0 * (u[+2, 0, 0] + u[-2, 0, 0])
            + 15.0 * (u[+1, 0, 0] + u[-1, 0, 0])
            - 20.0 * u[0, 0, 0]
        )
        adv_u_y = v[0, 0, 0] / (60.0 * dy) * (
            +45.0 * (u[0, +1, 0] - u[0, -1, 0])
            - 9.0 * (u[0, +2, 0] - u[0, -2, 0])
            + (u[0, +3, 0] - u[0, -3, 0])
        ) - abs_v[0, 0, 0] / (60.0 * dy) * (
            +(u[0, +3, 0] + u[0, -3, 0])
            - 6.0 * (u[0, +2, 0] + u[0, -2, 0])
            + 15.0 * (u[0, +1, 0] + u[0, -1, 0])
            - 20.0 * u[0, 0, 0]
        )
        adv_v_x = u[0, 0, 0] / (60.0 * dx) * (
            +45.0 * (v[+1, 0, 0] - v[-1, 0, 0])
            - 9.0 * (v[+2, 0, 0] - v[-2, 0, 0])
            + (v[+3, 0, 0] - v[-3, 0, 0])
        ) - abs_u[0, 0, 0] / (60.0 * dx) * (
            +(v[+3, 0, 0] + v[-3, 0, 0])
            - 6.0 * (v[+2, 0, 0] + v[-2, 0, 0])
            + 15.0 * (v[+1, 0, 0] + v[-1, 0, 0])
            - 20.0 * v[0, 0, 0]
        )
        adv_v_y = v[0, 0, 0] / (60.0 * dy) * (
            +45.0 * (v[0, +1, 0] - v[0, -1, 0])
            - 9.0 * (v[0, +2, 0] - v[0, -2, 0])
            + (v[0, +3, 0] - v[0, -3, 0])
        ) - abs_v[0, 0, 0] / (60.0 * dy) * (
            +(v[0, +3, 0] + v[0, -3, 0])
            - 6.0 * (v[0, +2, 0] + v[0, -2, 0])
            + 15.0 * (v[0, +1, 0] + v[0, -1, 0])
            - 20.0 * v[0, 0, 0]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _SixthOrder(BurgersAdvection):
    extent = 3

    @staticmethod
    @gtscript.function
    def __call__(dx, dy, u, v):
        adv_u_x = (
            u[0, 0, 0]
            / (60.0 * dx)
            * (
                +45.0 * (u[+1, 0, 0] - u[-1, 0, 0])
                - 9.0 * (u[+2, 0, 0] - u[-2, 0, 0])
                + (u[+3, 0, 0] - u[-3, 0, 0])
            )
        )
        adv_u_y = (
            v[0, 0, 0]
            / (60.0 * dy)
            * (
                +45.0 * (u[0, +1, 0] - u[0, -1, 0])
                - 9.0 * (u[0, +2, 0] - u[0, -2, 0])
                + (u[0, +3, 0] - u[0, -3, 0])
            )
        )
        adv_v_x = (
            u[0, 0, 0]
            / (60.0 * dx)
            * (
                +45.0 * (v[+1, 0, 0] - v[-1, 0, 0])
                - 9.0 * (v[+2, 0, 0] - v[-2, 0, 0])
                + (v[+3, 0, 0] - v[-3, 0, 0])
            )
        )
        adv_v_y = (
            v[0, 0, 0]
            / (60.0 * dy)
            * (
                +45.0 * (v[0, +1, 0] - v[0, -1, 0])
                - 9.0 * (v[0, +2, 0] - v[0, -2, 0])
                + (v[0, +3, 0] - v[0, -3, 0])
            )
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y
