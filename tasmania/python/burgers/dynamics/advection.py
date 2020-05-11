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
import numpy as np
from typing import Tuple

from gt4py import gtscript

from tasmania.python.utils import taz_types


class BurgersAdvection(abc.ABC):
    """ A discretizer for the 2-D Burgers advection flux. """

    extent: int = 0

    def __init__(self, gt_powered: bool) -> None:
        self.call = self.call_gt if gt_powered else self.call_numpy

    @staticmethod
    @abc.abstractmethod
    def call_numpy(
        dx: float, dy: float, u: np.ndarray, v: np.ndarray
    ) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """ Compute the accelerations due to advection.

        Vanilla NumPy implementation.

        Parameters
        ----------
        dx : float
            x-grid spacing.
        dy : float
            y-grid spacing.
        u : numpy.ndarray
            u-velocity.
        v : numpy.ndarray
            v-velocity.

        Return
        ------
        adv_u_x : numpy.ndarray
            x-acceleration for u-velocity.
        adv_u_y : numpy.ndarray
            y-acceleration for u-velocity.
        adv_v_x : numpy.ndarray
            x-acceleration for v-velocity.
        adv_v_y : numpy.ndarray
            y-acceleration for v-velocity.
        """
        pass

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def call_gt(
        dx: float, dy: float, u: taz_types.gtfield_t, v: taz_types.gtfield_t
    ) -> "Tuple[taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t]":
        """
        Compute the accelerations due to advection. GT4Py-based implementation.

        Parameters
        ----------
        dx : float
            x-grid spacing.
        dy : float
            y-grid spacing.
        u : gt4py.gtscript.Field
            u-velocity.
        v : gt4py.gtscript.Field
            v-velocity.

        Return
        ------
        adv_u_x : gt4py.gtscript.Field
            x-acceleration for u-velocity.
        adv_u_y : gt4py.gtscript.Field
            y-acceleration for u-velocity.
        adv_v_x : gt4py.gtscript.Field
            x-acceleration for v-velocity.
        adv_v_y : gt4py.gtscript.Field
            y-acceleration for v-velocity.
        """
        pass

    @staticmethod
    def factory(flux_scheme: str, gt_powered: bool) -> "BurgersAdvection":
        if flux_scheme == "first_order":
            return _FirstOrder(gt_powered)
        elif flux_scheme == "second_order":
            return _SecondOrder(gt_powered)
        elif flux_scheme == "third_order":
            return _ThirdOrder(gt_powered)
        elif flux_scheme == "fourth_order":
            return _FourthOrder(gt_powered)
        elif flux_scheme == "fifth_order":
            return _FifthOrder(gt_powered)
        elif flux_scheme == "sixth_order":
            return _SixthOrder(gt_powered)
        else:
            raise RuntimeError()


class _FirstOrder(BurgersAdvection):
    extent = 1

    def __init__(self, gt_powered):
        super().__init__(gt_powered)

    @staticmethod
    def call_numpy(dx, dy, u, v):
        abs_u = np.abs(u[1:-1, 1:-1])
        abs_v = np.abs(v[1:-1, 1:-1])

        adv_u_x = u[1:-1, 1:-1] / (2.0 * dx) * (u[2:, 1:-1] - u[:-2, 1:-1]) - abs_u / (
            2.0 * dx
        ) * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        adv_u_y = v[1:-1, 1:-1] / (2.0 * dy) * (u[1:-1, 2:] - u[1:-1, :-2]) - abs_v / (
            2.0 * dy
        ) * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
        adv_v_x = u[1:-1, 1:-1] / (2.0 * dx) * (v[2:, 1:-1] - v[:-2, 1:-1]) - abs_u / (
            2.0 * dx
        ) * (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1])
        adv_v_y = v[1:-1, 1:-1] / (2.0 * dy) * (v[1:-1, 2:] - v[1:-1, :-2]) - abs_v / (
            2.0 * dy
        ) * (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
        abs_u = u if u > 0 else -u
        abs_v = v if v > 0 else -v

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
    def call_numpy(dx, dy, u, v):
        adv_u_x = u[1:-1, 1:-1] / (2.0 * dx) * (u[2:, 1:-1] - u[:-2, 1:-1])
        adv_u_y = v[1:-1, 1:-1] / (2.0 * dy) * (u[1:-1, 2:] - u[1:-1, :-2])
        adv_v_x = u[1:-1, 1:-1] / (2.0 * dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
        adv_v_y = v[1:-1, 1:-1] / (2.0 * dy) * (v[1:-1, 2:] - v[1:-1, :-2])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
        adv_u_x = u[0, 0, 0] / (2.0 * dx) * (u[+1, 0, 0] - u[-1, 0, 0])
        adv_u_y = v[0, 0, 0] / (2.0 * dy) * (u[0, +1, 0] - u[0, -1, 0])
        adv_v_x = u[0, 0, 0] / (2.0 * dx) * (v[+1, 0, 0] - v[-1, 0, 0])
        adv_v_y = v[0, 0, 0] / (2.0 * dy) * (v[0, +1, 0] - v[0, -1, 0])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _ThirdOrder(BurgersAdvection):
    extent = 2

    @staticmethod
    def call_numpy(dx, dy, u, v):
        abs_u = np.abs(u[2:-2, 2:-2])
        abs_v = np.abs(v[2:-2, 2:-2])

        adv_u_x = u[2:-2, 2:-2] / (12.0 * dx) * (
            8.0 * (u[3:-1, 2:-2] - u[1:-3, 2:-2]) - (u[4:, 2:-2] - u[:-4, 2:-2])
        ) + abs_u / (12.0 * dx) * (
            u[4:, 2:-2]
            + u[:-4, 2:-2]
            - 4.0 * (u[3:-1, 2:-2] + u[1:-3, 2:-2])
            + 6.0 * u[2:-2, 2:-2]
        )
        adv_u_y = v[2:-2, 2:-2] / (12.0 * dy) * (
            8.0 * (u[2:-2, 3:-1] - u[2:-2, 1:-3]) - (u[2:-2, 4:] - u[2:-2, :-4])
        ) + abs_v / (12.0 * dy) * (
            u[2:-2, 4:]
            + u[2:-2, :-4]
            - 4.0 * (u[2:-2, 3:-1] + u[2:-2, 1:-3])
            + 6.0 * u[2:-2, 2:-2]
        )
        adv_v_x = u[2:-2, 2:-2] / (12.0 * dx) * (
            8.0 * (v[3:-1, 2:-2] - v[1:-3, 2:-2]) - (v[4:, 2:-2] - v[:-4, 2:-2])
        ) + abs_u / (12.0 * dx) * (
            v[4:, 2:-2]
            + v[:-4, 2:-2]
            - 4.0 * (v[3:-1, 2:-2] + v[1:-3, 2:-2])
            + 6.0 * v[2:-2, 2:-2]
        )
        adv_v_y = v[2:-2, 2:-2] / (12.0 * dy) * (
            8.0 * (v[2:-2, 3:-1] - v[2:-2, 1:-3]) - (v[2:-2, 4:] - v[2:-2, :-4])
        ) + abs_v / (12.0 * dy) * (
            v[2:-2, 4:]
            + v[2:-2, :-4]
            - 4.0 * (v[2:-2, 3:-1] + v[2:-2, 1:-3])
            + 6.0 * v[2:-2, 2:-2]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
        abs_u = u if u > 0 else -u
        abs_v = v if v > 0 else -v

        adv_u_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (u[+1, 0, 0] - u[-1, 0, 0]) - (u[+2, 0, 0] - u[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            u[+2, 0, 0]
            + u[-2, 0, 0]
            - 4.0 * (u[+1, 0, 0] + u[-1, 0, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_u_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (u[0, +1, 0] - u[0, -1, 0]) - (u[0, +2, 0] - u[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            u[0, +2, 0]
            + u[0, -2, 0]
            - 4.0 * (u[0, +1, 0] + u[0, -1, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_v_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (v[+1, 0, 0] - v[-1, 0, 0]) - (v[+2, 0, 0] - v[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            v[+2, 0, 0]
            + v[-2, 0, 0]
            - 4.0 * (v[+1, 0, 0] + v[-1, 0, 0])
            + 6.0 * v[0, 0, 0]
        )
        adv_v_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (v[0, +1, 0] - v[0, -1, 0]) - (v[0, +2, 0] - v[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            v[0, +2, 0]
            + v[0, -2, 0]
            - 4.0 * (v[0, +1, 0] + v[0, -1, 0])
            + 6.0 * v[0, 0, 0]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _FourthOrder(BurgersAdvection):
    extent = 2

    @staticmethod
    def call_numpy(dx, dy, u, v):
        adv_u_x = (
            u[2:-2, 2:-2]
            / (12.0 * dx)
            * (8.0 * (u[3:-1, 2:-2] - u[1:-3, 2:-2]) - (u[4:, 2:-2] - u[:-4, 2:-2]))
        )
        adv_u_y = (
            v[2:-2, 2:-2]
            / (12.0 * dy)
            * (8.0 * (u[2:-2, 3:-1] - u[2:-2, 1:-3]) - (u[2:-2, 4:] - u[2:-2, :-4]))
        )
        adv_v_x = (
            u[2:-2, 2:-2]
            / (12.0 * dx)
            * (8.0 * (v[3:-1, 2:-2] - v[1:-3, 2:-2]) - (v[4:, 2:-2] - v[:-4, 2:-2]))
        )
        adv_v_y = (
            v[2:-2, 2:-2]
            / (12.0 * dy)
            * (8.0 * (v[2:-2, 3:-1] - v[2:-2, 1:-3]) - (v[2:-2, 4:] - v[2:-2, :-4]))
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
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
    def call_numpy(dx, dy, u, v):
        abs_u = np.abs(u[3:-3, 3:-3])
        abs_v = np.abs(v[3:-3, 3:-3])

        adv_u_x = u[3:-3, 3:-3] / (60.0 * dx) * (
            +45.0 * (u[4:-2, 3:-3] - u[2:-4, 3:-3])
            - 9.0 * (u[5:-1, 3:-3] - u[1:-5, 3:-3])
            + (u[6:, 3:-3] - u[:-6, 3:-3])
        ) - abs_u / (60.0 * dx) * (
            +(u[6:, 3:-3] + u[:-6, 3:-3])
            - 6.0 * (u[5:-1, 3:-3] + u[1:-5, 3:-3])
            + 15.0 * (u[4:-2, 3:-3] + u[2:-4, 3:-3])
            - 20.0 * u[3:-3, 3:-3]
        )
        adv_u_y = v[3:-3, 3:-3] / (60.0 * dy) * (
            +45.0 * (u[3:-3, 4:-2] - u[3:-3, 2:-4])
            - 9.0 * (u[3:-3, 5:-1] - u[3:-3, 1:-5])
            + (u[3:-3, 6:] - u[3:-3, :-6])
        ) - abs_v / (60.0 * dy) * (
            +(u[3:-3, 6:] + u[3:-3, :-6])
            - 6.0 * (u[3:-3, 5:-1] + u[3:-3, 1:-5])
            + 15.0 * (u[3:-3, 4:-2] + u[3:-3, 2:-4])
            - 20.0 * u[3:-3, 3:-3]
        )
        adv_v_x = u[3:-3, 3:-3] / (60.0 * dx) * (
            +45.0 * (v[4:-2, 3:-3] - v[2:-4, 3:-3])
            - 9.0 * (v[5:-1, 3:-3] - v[1:-5, 3:-3])
            + (v[6:, 3:-3] - v[:-6, 3:-3])
        ) - abs_u / (60.0 * dx) * (
            +(v[6:, 3:-3] + v[:-6, 3:-3])
            - 6.0 * (v[5:-1, 3:-3] + v[1:-5, 3:-3])
            + 15.0 * (v[4:-2, 3:-3] + v[2:-4, 3:-3])
            - 20.0 * v[3:-3, 3:-3]
        )
        adv_v_y = v[3:-3, 3:-3] / (60.0 * dy) * (
            +45.0 * (v[3:-3, 4:-2] - v[3:-3, 2:-4])
            - 9.0 * (v[3:-3, 5:-1] - v[3:-3, 1:-5])
            + (v[3:-3, 6:] - v[3:-3, :-6])
        ) - abs_v / (60.0 * dy) * (
            +(v[3:-3, 6:] + v[3:-3, :-6])
            - 6.0 * (v[3:-3, 5:-1] + v[3:-3, 1:-5])
            + 15.0 * (v[3:-3, 4:-2] + v[3:-3, 2:-4])
            - 20.0 * v[3:-3, 3:-3]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
        abs_u = u if u > 0 else -u
        abs_v = v if v > 0 else -v

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
    def call_numpy(dx, dy, u, v):
        adv_u_x = (
            u[3:-3, 3:-3]
            / (60.0 * dx)
            * (
                +45.0 * (u[4:-2, 3:-3] - u[2:-4, 3:-3])
                - 9.0 * (u[5:-1, 3:-3] - u[1:-5, 3:-3])
                + (u[6:, 3:-3] - u[:-6, 3:-3])
            )
        )
        adv_u_y = (
            v[3:-3, 3:-3]
            / (60.0 * dy)
            * (
                +45.0 * (u[3:-3, 4:-2] - u[3:-3, 2:-4])
                - 9.0 * (u[3:-3, 5:-1] - u[3:-3, 1:-5])
                + (u[3:-3, 6:] - u[3:-3, :-6])
            )
        )
        adv_v_x = (
            u[3:-3, 3:-3]
            / (60.0 * dx)
            * (
                +45.0 * (v[4:-2, 3:-3] - v[2:-4, 3:-3])
                - 9.0 * (v[5:-1, 3:-3] - v[1:-5, 3:-3])
                + (v[6:, 3:-3] - v[:-6, 3:-3])
            )
        )
        adv_v_y = (
            v[3:-3, 3:-3]
            / (60.0 * dy)
            * (
                +45.0 * (v[3:-3, 4:-2] - v[3:-3, 2:-4])
                - 9.0 * (v[3:-3, 5:-1] - v[3:-3, 1:-5])
                + (v[3:-3, 6:] - v[3:-3, :-6])
            )
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @gtscript.function
    def call_gt(dx, dy, u, v):
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
