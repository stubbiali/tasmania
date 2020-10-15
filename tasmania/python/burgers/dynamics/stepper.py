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
from typing import Optional, TYPE_CHECKING

from gt4py import gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt

if TYPE_CHECKING:
    from tasmania.python.domain.horizontal_grid import HorizontalGrid


class ForwardEulerStepNumpy:
    def __init__(self, advection):
        self.advection = advection

    def __call__(
        self,
        in_u: np.ndarray,
        in_v: np.ndarray,
        in_u_tmp: np.ndarray,
        in_v_tmp: np.ndarray,
        out_u: np.ndarray,
        out_v: np.ndarray,
        in_u_tnd: np.ndarray = None,
        in_v_tnd: np.ndarray = None,
        *,
        dt: float,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        nb = self.advection.extent

        istart, istop = origin[0], origin[0] + domain[0]
        i = slice(istart, istop)
        iext = slice(istart - nb, istop + nb)
        jstart, jstop = origin[1], origin[1] + domain[1]
        j = slice(jstart, jstop)
        jext = slice(jstart - nb, jstop + nb)
        kstart, kstop = origin[2], origin[2] + domain[2]
        k = slice(kstart, kstop)

        adv_u_x, adv_u_y, adv_v_x, adv_v_y = self.advection.call_numpy(
            dx=dx, dy=dy, u=in_u_tmp[iext, jext, k], v=in_v_tmp[iext, jext, k]
        )

        if in_u_tnd is not None:
            out_u[i, j, k] = in_u[i, j, k] - dt * (
                adv_u_x + adv_u_y - in_u_tnd[i, j, k]
            )
        else:
            out_u[i, j, k] = in_u[i, j, k] - dt * (adv_u_x + adv_u_y)

        if in_v_tnd is not None:
            out_v[i, j, k] = in_v[i, j, k] - dt * (
                adv_v_x + adv_v_y - in_v_tnd[i, j, k]
            )
        else:
            out_v[i, j, k] = in_v[i, j, k] - dt * (adv_v_x + adv_v_y)


def forward_euler_step_gt(
    in_u: gtscript.Field["dtype"],
    in_v: gtscript.Field["dtype"],
    in_u_tmp: gtscript.Field["dtype"],
    in_v_tmp: gtscript.Field["dtype"],
    out_u: gtscript.Field["dtype"],
    out_v: gtscript.Field["dtype"],
    in_u_tnd: gtscript.Field["dtype"] = None,
    in_v_tnd: gtscript.Field["dtype"] = None,
    *,
    dt: float,
    dx: float,
    dy: float
) -> None:
    from __externals__ import advection, tnd_u, tnd_v

    with computation(PARALLEL), interval(...):
        adv_u_x, adv_u_y, adv_v_x, adv_v_y = advection(
            dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp
        )

        if __INLINED(tnd_u):
            out_u = in_u[0, 0, 0] - dt * (
                adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0] - in_u_tnd[0, 0, 0]
            )
        else:
            out_u = in_u[0, 0, 0] - dt * (adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0])

        if __INLINED(tnd_v):
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
        grid_xy: "HorizontalGrid",
        nb: int,
        flux_scheme: str,
        backend: str,
        backend_opts: taz_types.options_dict_t,
        dtype: taz_types.dtype_t,
        build_info: taz_types.options_dict_t,
        exec_info: taz_types.mutable_options_dict_t,
        default_origin: taz_types.triplet_int_t,
        rebuild: bool,
        managed_memory: bool,
    ) -> None:
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
        backend : str
            The backend.
        backend_opts : dict
            Dictionary of backend-specific options.
        build_info : dict
            Dictionary of building options.
        dtype : data-type
            Data type of the storages.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        managed_memory : bool
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
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

        self._advection = BurgersAdvection.factory(flux_scheme, backend)

        assert nb >= self._advection.extent
        self._nb = nb

        self._stencil = None

    @property
    @abc.abstractmethod
    def stages(self) -> int:
        """
        Returns
        -------
        int :
            Number of stages the time integrator consists of.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self,
        stage: int,
        state: taz_types.gtstorage_dict_t,
        tendencies: taz_types.gtstorage_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.gtstorage_dict_t:
        """
        Performing a stage of the time integrator.

        Parameters
        ----------
        stage : int
            The stage to be performed.
        state : dict[str, gt4py.storage.storage.Storage]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`gt4py.storage.storage.Storage`\s
            storing values for those variables.
        tendencies : dict[str, gt4py.storage.storage.Storage]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`gt4py.storage.storage.Storage`\s
            storing tendencies for those variables.
        timestep : datetime.timedelta
            The time step size.

        Return
        ------
        dict[str, gt4py.storage.storage.Storage]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`gt4py.storage.storage.Storage`\s
            storing new values for those variables.
        """
        pass

    @staticmethod
    def factory(
        time_integration_scheme: str,
        grid_xy: "HorizontalGrid",
        nb: int,
        flux_scheme: str,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        managed_memory: bool = False
    ) -> "BurgersStepper":
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
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.

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
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )
        if time_integration_scheme == "forward_euler":
            return ForwardEuler(*args)
        elif time_integration_scheme == "rk2":
            return RK2(*args)
        elif time_integration_scheme == "rk3ws":
            return RK3WS(*args)
        else:
            raise RuntimeError()


class ForwardEuler(BurgersStepper):
    """ The forward Euler time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        dtype,
        build_info,
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
            dtype,
            build_info,
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
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info,
            validate_args=False
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
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
            "out_v": zeros(
                storage_shape,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
        }

        if is_gt(backend):
            self._stencil = gtscript.stencil(
                definition=forward_euler_step_gt,
                name=self.__class__.__name__,
                backend=get_gt_backend(backend),
                build_info=self._build_info,
                rebuild=self._rebuild,
                dtypes={"dtype": dtype},
                externals={
                    "advection": self._advection.call_gt,
                    "tnd_u": "x_velocity" in tendencies,
                    "tnd_v": "y_velocity" in tendencies,
                },
                **(self._backend_opts or {})
            )
        else:
            self._stencil = ForwardEulerStepNumpy(self._advection)


class RK2(BurgersStepper):
    """ A two-stages Runge-Kutta time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        dtype,
        build_info,
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
            dtype,
            build_info,
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
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info,
            validate_args=False
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
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
            "out_v": zeros(
                storage_shape,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            ),
        }

        if is_gt(backend):
            self._stencil = gtscript.stencil(
                definition=forward_euler_step_gt,
                name=self.__class__.__name__,
                backend=get_gt_backend(backend),
                build_info=self._build_info,
                rebuild=self._rebuild,
                dtypes={"dtype": dtype},
                externals={
                    "advection": self._advection.call_gt,
                    "tnd_u": "x_velocity" in tendencies,
                    "tnd_v": "y_velocity" in tendencies,
                },
                **(self._backend_opts or {})
            )
        else:
            self._stencil = ForwardEulerStepNumpy(self._advection)


class RK3WS(RK2):
    """ A three-stages Runge-Kutta time integrator. """

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_opts,
        dtype,
        build_info,
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
            dtype,
            build_info,
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
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self._exec_info,
            validate_args=False
        )

        return {
            "time": state["time"] + dtr,
            "x_velocity": self._stencil_args["out_u"],
            "y_velocity": self._stencil_args["out_v"],
        }
