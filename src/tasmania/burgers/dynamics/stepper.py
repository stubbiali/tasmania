# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from __future__ import annotations
import abc
import numpy as np
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript

from tasmania.burgers.dynamics.advection import BurgersAdvection
from tasmania.externals import numba
from tasmania.framework.register import factorize
from tasmania.framework.stencil import StencilFactory
from tasmania.framework.tag import stencil_definition

if TYPE_CHECKING:
    from typing import Optional

    from tasmania.domain.horizontal_grid import HorizontalGrid
    from tasmania.framework.options import BackendOptions, StorageOptions
    from tasmania.utils.typingx import NDArrayDict, TimeDelta, TripletInt


class BurgersStepper(StencilFactory, abc.ABC):
    """
    Abstract base class whose children integrate the 2-D inviscid Burgers
    equations implementing different time integrators.
    """

    registry = {}

    def __init__(
        self,
        grid_xy: HorizontalGrid,
        nb: int,
        flux_scheme: str,
        backend: str,
        backend_options: BackendOptions,
        storage_options: StorageOptions,
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
        backend_options : BackendOptions
            Backend-specific options.
        storage_options : StorageOptions
            Storage-related options.
        """
        super().__init__(backend, backend_options, storage_options)
        self._grid_xy = grid_xy
        self._advection = BurgersAdvection.factory(flux_scheme, backend)
        assert nb >= self._advection.extent
        self._nb = nb
        self._forward_euler = None

    @property
    @abc.abstractmethod
    def stages(self) -> int:
        """
        Returns
        -------
        int :
            Number of stages the time integrator consists of.
        """

    @abc.abstractmethod
    def __call__(
        self,
        stage: int,
        state: NDArrayDict,
        tendencies: NDArrayDict,
        timestep: TimeDelta,
        out_state: NDArrayDict,
    ) -> None:
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

    @staticmethod
    def factory(
        time_integration_scheme: str,
        grid_xy: HorizontalGrid,
        nb: int,
        flux_scheme: str,
        *,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> "BurgersStepper":
        """
        Parameters
        ----------
        time_integration_scheme : str
            String specifying the time integrator to be used.
        grid_xy : tasmania.HorizontalGrid
            The underlying horizontal grid.
        nb : int
            Number of boundary layers.
        flux_scheme : str
            String specifying the advective flux scheme to be used.
            See :class:`tasmania.BurgersAdvection` for all available options.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.

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
            backend_options,
            storage_options,
        )
        return factorize(time_integration_scheme, BurgersStepper, args)

    def _stencil_initialize(self, tendencies: NDArrayDict) -> None:
        self._stencil_args = {}
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "advection": self._advection.get_subroutine_definition("advection"),
            "extent": self._advection.extent,
            "tnd_u": "x_velocity" in tendencies,
            "tnd_v": "y_velocity" in tendencies,
        }
        self._forward_euler = self.compile_stencil("forward_euler")

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="forward_euler")
    def _forward_euler_numpy(
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
        origin: TripletInt,
        domain: TripletInt,
    ) -> None:
        istart, istop = origin[0], origin[0] + domain[0]
        i = slice(istart, istop)
        iext = slice(istart - extent, istop + extent)
        jstart, jstop = origin[1], origin[1] + domain[1]
        j = slice(jstart, jstop)
        jext = slice(jstart - extent, jstop + extent)
        kstart, kstop = origin[2], origin[2] + domain[2]
        k = slice(kstart, kstop)

        adv_u_x, adv_u_y, adv_v_x, adv_v_y = advection(
            dx=dx, dy=dy, u=in_u_tmp[iext, jext, k], v=in_v_tmp[iext, jext, k]
        )

        if in_u_tnd:
            out_u[i, j, k] = in_u[i, j, k] - dt * (adv_u_x + adv_u_y - in_u_tnd[i, j, k])
        else:
            out_u[i, j, k] = in_u[i, j, k] - dt * (adv_u_x + adv_u_y)

        if in_v_tnd:
            out_v[i, j, k] = in_v[i, j, k] - dt * (adv_v_x + adv_v_y - in_v_tnd[i, j, k])
        else:
            out_v[i, j, k] = in_v[i, j, k] - dt * (adv_v_x + adv_v_y)

    @staticmethod
    @stencil_definition(backend="gt4pt*", stencil="forward_euler")
    def _forward_euler_gt4py(
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
        dy: float,
    ) -> None:
        from __externals__ import advection, tnd_u, tnd_v

        with computation(PARALLEL), interval(...):
            adv_u_x, adv_u_y, adv_v_x, adv_v_y = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)

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

    if numba:

        @staticmethod
        @stencil_definition(backend="numba:cpu:stencil", stencil="forward_euler")
        def _forward_euler_numba_cpu(
            in_u: np.ndarray,
            in_v: np.ndarray,
            in_u_tmp: np.ndarray,
            in_v_tmp: np.ndarray,
            out_u: np.ndarray,
            out_v: np.ndarray,
            in_u_tnd: Optional[np.ndarray] = None,
            in_v_tnd: Optional[np.ndarray] = None,
            *,
            dt: float,
            dx: float,
            dy: float,
            origin: TripletInt,
            domain: TripletInt,
        ) -> None:
            # >>> stencil definitions
            def step_def(phi, adv_x, adv_y, dt):
                return phi[0, 0, 0] - dt * (adv_x[0, 0, 0] + adv_y[0, 0, 0])

            def step_tnd_def(phi, adv_x, adv_y, tnd, dt):
                return phi[0, 0, 0] - dt * (adv_x[0, 0, 0] + adv_y[0, 0, 0] - tnd[0, 0, 0])

            # >>> stencil compilations
            step = numba.stencil(step_def)
            step_tnd = numba.stencil(step_tnd_def)

            # >>> calculations
            ib, jb, kb = origin
            ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]

            adv_u_x, adv_u_y, adv_v_x, adv_v_y = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)

            if in_u_tnd:
                step_tnd(
                    in_u[ib:ie, jb:je, kb:ke],
                    adv_u_x[ib:ie, jb:je, kb:ke],
                    adv_u_y[ib:ie, jb:je, kb:ke],
                    in_u_tnd[ib:ie, jb:je, kb:ke],
                    dt,
                    out=out_u[ib:ie, jb:je, kb:ke],
                )
            else:
                step(
                    in_u[ib:ie, jb:je, kb:ke],
                    adv_u_x[ib:ie, jb:je, kb:ke],
                    adv_u_y[ib:ie, jb:je, kb:ke],
                    dt,
                    out=out_u[ib:ie, jb:je, kb:ke],
                )

            if in_v_tnd:
                step_tnd(
                    in_v[ib:ie, jb:je, kb:ke],
                    adv_v_x[ib:ie, jb:je, kb:ke],
                    adv_v_y[ib:ie, jb:je, kb:ke],
                    in_v_tnd[ib:ie, jb:je, kb:ke],
                    dt,
                    out=out_v[ib:ie, jb:je, kb:ke],
                )
            else:
                step(
                    in_v[ib:ie, jb:je, kb:ke],
                    adv_v_x[ib:ie, jb:je, kb:ke],
                    adv_v_y[ib:ie, jb:je, kb:ke],
                    dt,
                    out=out_v[ib:ie, jb:je, kb:ke],
                )
