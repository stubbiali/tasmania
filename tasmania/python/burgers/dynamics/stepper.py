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

from gridtools import __externals__
from gridtools import gtscript

# from gridtools.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import nb as conf_nb
except ImportError:
    conf_nb = None

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


def forward_euler_step(
    in_u: gtscript.Field[np.float64],
    in_v: gtscript.Field[np.float64],
    in_u_tmp: gtscript.Field[np.float64],
    in_v_tmp: gtscript.Field[np.float64],
    out_u: gtscript.Field[np.float64],
    out_v: gtscript.Field[np.float64],
    in_u_tnd: gtscript.Field[np.float64] = None,
    in_v_tnd: gtscript.Field[np.float64] = None,
    *,
    dt: float,
    dx: float,
    dy: float
):
    from __externals__ import advection, tnd_u, tnd_v

    with computation(PARALLEL), interval(...):
        adv_u_x, adv_u_y, adv_v_x, adv_v_y = advection(
            dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp
        )

        if tnd_u:
            out_u = in_u[0, 0, 0] - dt * (
                adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0] - in_u_tnd[0, 0, 0]
            )
        else:
            out_u = in_u[0, 0, 0] - dt * (adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0])

        if tnd_v:
            out_v = in_v[0, 0, 0] - dt * (
                adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0] - in_v_tnd[0, 0, 0]
            )
        else:
            out_v = in_v[0, 0, 0] - dt * (adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0])


class BurgersStepper(abc.ABC):
    """
    Abstract base class whose children integrate the 2-D inviscid Burgers
    equations implementing different time integrators.
    """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        """
        Parameters
        ----------
        grid_xy : tasmania.HorizontalGrid
            The underlying horizontal grid.
        nb : int
            Number of boundary layers.
        flux_scheme : str
            String specifying the advective flux scheme to be used.
            See :class:`tasmania.BurgersAdvection` for all available options.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `numpy.dtype`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        """
        self._grid_xy = grid_xy
        self._backend = backend
        self._backend_opts = backend_opts
        self._build_info = build_info
        self._dtype = dtype
        self._exec_info = exec_info
        self._default_origin = default_origin
        self._rebuild = rebuild
        self._managed_memory = managed_memory

        self._advection = BurgersAdvection.factory(flux_scheme)

        assert nb >= self._advection.extent
        self._nb = nb

        self._stencil = None

    @property
    @abc.abstractmethod
    def stages(self):
        """
        Returns
        -------
        int :
            Number of stages the time integrator consists of.
        """
        pass

    @abc.abstractmethod
    def __call__(self, stage, state, tendencies, timestep):
        """
        Performing a stage of the time integrator.

        Parameters
        ----------
        stage : int
            The stage to be performed.
        state : dict
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`numpy.ndarray`\s storing values
            for those variables.
        tendencies : dict
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`numpy.ndarray`\s storing tendencies
            for those variables.
        timestep : timedelta
            The time step size.

        Return
        ------
        dict :
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`numpy.ndarray`\s storing new values
            for those variables.
        """
        pass

    @staticmethod
    def factory(
        time_integration_scheme,
        grid_xy,
        nb,
        flux_scheme,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        managed_memory=False
    ):
        """
        Parameters
        ----------
        time_integration_scheme : str
            String specifying the time integrator to be used. Either:

                * 'forward_euler' for the forward Euler method;
                * 'rk2' for the explicit, two-stages, second-order \
                    Runge-Kutta (RK) method;
                * 'rk3ws' for the explicit, three-stages, second-order \
                    RK method by Wicker & Skamarock.
        grid_xy : tasmania.HorizontalGrid
            The underlying horizontal grid.
        nb : int
            Number of boundary layers.
        flux_scheme : str
            String specifying the advective flux scheme to be used.
            See :class:`tasmania.BurgersAdvection` for all available options.
        backend : `str`, optional
            TODO
        backend_opts : `dict`, optional
            TODO
        build_info : `dict`, optional
            TODO
        dtype : `numpy.dtype`, optional
            TODO
        exec_info : `dict`, optional
            TODO
        default_origin : `tuple`, optional
            TODO
        rebuild : `bool`, optional
            TODO
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.

        Return
        ------
        tasmania.BurgersStepper :
            An instance of the appropriate derived class.
        """
        args = (
            grid_xy,
            nb,
            flux_scheme,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )
        if time_integration_scheme == "forward_euler":
            return _ForwardEuler(*args)
        elif time_integration_scheme == "rk2":
            return _RK2(*args)
        elif time_integration_scheme == "rk3ws":
            return _RK3WS(*args)
        else:
            raise RuntimeError()


class _ForwardEuler(BurgersStepper):
    """ The forward Euler time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        super().__init__(
            grid_xy,
            nb,
            flux_scheme,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    @property
    def stages(self):
        return 1

    def __call__(self, stage, state, tendencies, timestep):
        nx, ny = self._grid_xy.nx, self._grid_xy.ny
        nb = self._nb

        if self._stencil is None:
            self._stencil_initialize(tendencies)

        dt = timestep.total_seconds()
        dx = self._grid_xy.dx.to_units("m").values.item()
        dy = self._grid_xy.dy.to_units("m").values.item()

        stencil_args = {}
        self._stencil_args["in_u"] = state["x_velocity"]
        self._stencil_args["in_u_tmp"] = state["x_velocity"]
        self._stencil_args["in_v"] = state["y_velocity"]
        self._stencil_args["in_v_tmp"] = state["y_velocity"]
        if "x_velocity" in tendencies:
            self._stencil_args["in_u_tnd"] = tendencies["x_velocity"]
        if "y_velocity" in tendencies:
            self._stencil_args["in_v_tnd"] = tendencies["y_velocity"]

        self._stencil(
            **self._stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info
        )

        return {
            "time": state["time"] + timestep,
            "x_velocity": self._stencil_args["out_u"],
            "y_velocity": self._stencil_args["out_v"],
        }

    def _stencil_initialize(self, tendencies):
        storage_shape = (self._grid_xy.nx, self._grid_xy.ny, 1)
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        managed_memory = self._managed_memory

        self._stencil_args = {
            "out_u": zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
            "out_v": zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
        }

        self._stencil = gtscript.stencil(
            definition=forward_euler_step,
            name=self.__class__.__name__,
            backend=backend,
            build_info=self._build_info,
            rebuild=self._rebuild,
            externals={
                "advection": self._advection.__call__,
                "tnd_u": "x_velocity" in tendencies,
                "tnd_v": "y_velocity" in tendencies,
            },
            **(self._backend_opts or {})
        )


class _RK2(BurgersStepper):
    """ A two-stages Runge-Kutta time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        super().__init__(
            grid_xy,
            nb,
            flux_scheme,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    @property
    def stages(self):
        return 2

    def __call__(self, stage, state, tendencies, timestep):
        nx, ny = self._grid_xy.nx, self._grid_xy.ny
        nb = self._nb

        if self._stencil is None:
            self._stencil_initialize(tendencies)

        dx = self._grid_xy.dx.to_units("m").values.item()
        dy = self._grid_xy.dy.to_units("m").values.item()

        if stage == 0:
            dt = 0.5 * timestep.total_seconds()
            self._stencil_args["in_u"] = state["x_velocity"]
            self._stencil_args["in_v"] = state["y_velocity"]
        else:
            dt = timestep.total_seconds()

        self._stencil_args["in_u_tmp"] = state["x_velocity"]
        self._stencil_args["in_v_tmp"] = state["y_velocity"]
        if "x_velocity" in tendencies:
            self._stencil_args["in_u_tnd"] = tendencies["x_velocity"]
        if "y_velocity" in tendencies:
            self._stencil_args["in_v_tnd"] = tendencies["y_velocity"]

        self._stencil(
            **self._stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info
        )

        return {
            "time": state["time"] + 0.5 * timestep,
            "x_velocity": self._stencil_args["out_u"],
            "y_velocity": self._stencil_args["out_v"],
        }

    def _stencil_initialize(self, tendencies):
        storage_shape = (self._grid_xy.nx, self._grid_xy.ny, 1)
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        managed_memory = self._managed_memory

        self._stencil_args = {
            "out_u": zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
            "out_v": zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
        }

        self._stencil = gtscript.stencil(
            definition=forward_euler_step,
            name=self.__class__.__name__,
            backend=backend,
            build_info=self._build_info,
            rebuild=self._rebuild,
            externals={
                "advection": self._advection.__call__,
                "tnd_u": "x_velocity" in tendencies,
                "tnd_v": "y_velocity" in tendencies,
            },
            **(self._backend_opts or {})
        )


class _RK3WS(_RK2):
    """ A three-stages Runge-Kutta time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        super().__init__(
            grid_xy,
            nb,
            flux_scheme,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    @property
    def stages(self):
        return 3

    def __call__(self, stage, state, tendencies, timestep):
        nx, ny = self._grid_xy.nx, self._grid_xy.ny
        nb = self._nb

        if self._stencil is None:
            self._stencil_initialize(tendencies)

        dx = self._grid_xy.dx.to_units("m").values.item()
        dy = self._grid_xy.dy.to_units("m").values.item()

        if stage == 0:
            dtr = 1.0 / 3.0 * timestep
            dt = 1.0 / 3.0 * timestep.total_seconds()
            self._stencil_args["in_u"] = state["x_velocity"]
            self._stencil_args["in_v"] = state["y_velocity"]
        elif stage == 1:
            dtr = 1.0 / 6.0 * timestep
            dt = 0.5 * timestep.total_seconds()
        else:
            dtr = 1.0 / 2.0 * timestep
            dt = timestep.total_seconds()

        self._stencil_args["in_u_tmp"] = state["x_velocity"]
        self._stencil_args["in_v_tmp"] = state["y_velocity"]
        if "x_velocity" in tendencies:
            self._stencil_args["in_u_tnd"] = tendencies["x_velocity"]
        if "y_velocity" in tendencies:
            self._stencil_args["in_v_tnd"] = tendencies["y_velocity"]

        self._stencil(
            **self._stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info
        )

        return {
            "time": state["time"] + dtr,
            "x_velocity": self._stencil_args["out_u"],
            "y_velocity": self._stencil_args["out_v"],
        }
