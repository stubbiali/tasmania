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
import numpy as np
from typing import Optional, TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.base_components import GridComponent
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.utils import typing
from tasmania.python.utils.gtscript import positive

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class HorizontalVelocity(GridComponent, StencilFactory):
    """
    This class diagnoses the horizontal momenta (respectively, velocity
    components) with the help of the air density and the horizontal
    velocity components (resp., momenta).
    """

    def __init__(
        self: "HorizontalVelocity",
        grid: "Grid",
        staggering: bool = True,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        staggering : `bool`, optional
            ``True`` if the velocity components should be computed
            on the staggered grid, ``False`` to collocate the velocity
            components in the mass points.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(grid)
        super(GridComponent, self).__init__(
            backend, backend_options, storage_options
        )

        # store input arguments needed at run-time
        self._staggering = staggering

        # initialize the underlying stencils
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {"staggering": staggering}
        self._stencil_diagnosing_momenta = self.compile("momenta")
        self._stencil_diagnosing_velocity_x = self.compile("velocity_x")
        self._stencil_diagnosing_velocity_y = self.compile("velocity_y")

    def get_momenta(
        self: "HorizontalVelocity",
        d: typing.array_t,
        u: typing.array_t,
        v: typing.array_t,
        du: typing.array_t,
        dv: typing.array_t,
    ) -> None:
        """
        Diagnose the horizontal momenta.

        Parameters
        ----------
        d : array-like
            The air density.
        u : array-like
            The x-velocity field.
        v : array-like
            The y-velocity field.
        du : array-like
            The buffer where the x-momentum will be written.
        dv : array-like
            The buffer where the y-momentum will be written.
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # run the stencil
        self._stencil_diagnosing_momenta(
            in_d=d,
            in_u=u,
            in_v=v,
            out_du=du,
            out_dv=dv,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    def get_velocity_components(
        self: "HorizontalVelocity",
        d: typing.array_t,
        du: typing.array_t,
        dv: typing.array_t,
        u: typing.array_t,
        v: typing.array_t,
    ) -> None:
        """
        Diagnose the horizontal velocity components.

        Parameters
        ----------
        d : array-like
            The air density.
        du : array-like
            The x-momentum.
        dv : array-like
            The y-momentum.
        u : array-like
            The buffer where the x-velocity will be written.
        v : array-like
            The buffer where the y-velocity will be written.

        Note
        ----
        If staggering is enabled, the first and last rows (respectively, columns)
        of the x-velocity (resp., y-velocity) are not set.
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dn = int(self._staggering)

        # run the stencils
        self._stencil_diagnosing_velocity_x(
            in_d=d,
            in_du=du,
            out_u=u,
            origin=(dn, 0, 0),
            domain=(nx - dn, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_diagnosing_velocity_y(
            in_d=d,
            in_dv=dv,
            out_v=v,
            origin=(0, dn, 0),
            domain=(nx, ny - dn, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    @stencil_definition(backend=("numpy", "cupy"), stencil="momenta")
    def _diagnose_momenta_numpy(
        self: "HorizontalVelocity",
        in_d: np.ndarray,
        in_u: np.ndarray,
        in_v: np.ndarray,
        out_du: np.ndarray,
        out_dv: np.ndarray,
        *,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        j = slice(origin[1], origin[1] + domain[1])
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        k = slice(origin[2], origin[2] + domain[2])

        if self._staggering:  # compile-time if
            out_du[i, j, k] = (
                0.5 * in_d[i, j, k] * (in_u[i, j, k] + in_u[ip1, j, k])
            )
            out_dv[i, j, k] = (
                0.5 * in_d[i, j, k] * (in_v[i, j, k] + in_v[i, jp1, k])
            )
        else:
            out_du[i, j, k] = in_d[i, j, k] * in_u[i, j, k]
            out_dv[i, j, k] = in_d[i, j, k] * in_v[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="momenta")
    def _diagnose_momenta_gt4py(
        in_d: gtscript.Field["dtype"],
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_du: gtscript.Field["dtype"],
        out_dv: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if __INLINED(staggering):  # compile-time if
                out_du = 0.5 * in_d[0, 0, 0] * (in_u[0, 0, 0] + in_u[1, 0, 0])
                out_dv = 0.5 * in_d[0, 0, 0] * (in_v[0, 0, 0] + in_v[0, 1, 0])
            else:
                out_du = in_d * in_u
                out_dv = in_d * in_v

    @stencil_definition(backend=("numpy", "cupy"), stencil="velocity_x")
    def _diagnose_velocity_x_numpy(
        self: "HorizontalVelocity",
        in_d: np.ndarray,
        in_du: np.ndarray,
        out_u: np.ndarray,
        *,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        if self._staggering:  # compile-time if
            out_u[i, j, k] = (in_du[im1, j, k] + in_du[i, j, k]) / (
                in_d[im1, j, k] + in_d[i, j, k]
            )
        else:
            out_u[i, j, k] = in_du[i, j, k] / in_d[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="velocity_x")
    def _diagnose_velocity_x_gt4py(
        in_d: gtscript.Field["dtype"],
        in_du: gtscript.Field["dtype"],
        out_u: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if __INLINED(staggering):  # compile-time if
                out_u = (in_du[-1, 0, 0] + in_du[0, 0, 0]) / (
                    in_d[-1, 0, 0] + in_d[0, 0, 0]
                )
            else:
                out_u = in_du / in_d

    @stencil_definition(backend=("numpy", "cupy"), stencil="velocity_y")
    def _diagnose_velocity_y_numpy(
        self: "HorizontalVelocity",
        in_d: np.ndarray,
        in_dv: np.ndarray,
        out_v: np.ndarray,
        *,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        k = slice(origin[2], origin[2] + domain[2])

        if self._staggering:  # compile-time if
            out_v[i, j, k] = (in_dv[i, jm1, k] + in_dv[i, j, k]) / (
                in_d[i, jm1, k] + in_d[i, j, k]
            )
        else:
            out_v[i, j, k] = in_dv[i, j, k] / in_d[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="velocity_y")
    def _diagnose_velocity_y_gt4py(
        in_d: gtscript.Field["dtype"],
        in_dv: gtscript.Field["dtype"],
        out_v: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if __INLINED(staggering):  # compile-time if
                out_v = (in_dv[0, -1, 0] + in_dv[0, 0, 0]) / (
                    in_d[0, -1, 0] + in_d[0, 0, 0]
                )
            else:
                out_v = in_dv / in_d


class WaterConstituent(GridComponent, StencilFactory):
    """
    This class diagnoses the density (respectively, mass fraction) of any water
    constituent with the help of the air density and the mass fraction (resp.,
    the density) of that water constituent.
    """

    def __init__(
        self: "WaterConstituent",
        grid: "Grid",
        clipping: bool = False,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        clipping : `bool`, optional
            ``True`` to clip the negative values of the output fields,
            ``False`` otherwise. Defaults to ``False``.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options
        """
        super().__init__(grid)
        super(GridComponent, self).__init__(
            backend, backend_options, storage_options
        )

        # store input arguments needed at run-time
        self._clipping = clipping

        # initialize the underlying stencils
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "clipping": clipping,
            "positive": positive,
        }
        self._stencil_diagnosing_density = self.compile("density")
        self._stencil_diagnosing_mass_fraction = self.compile("mass_fraction")

    def get_density_of_water_constituent(
        self: "WaterConstituent",
        d: typing.array_t,
        q: typing.array_t,
        dq: typing.array_t,
    ) -> None:
        """
        Diagnose the density of a water constituent.

        Parameters
        ----------
        d : array-like
            The air density.
        q : array-like
            The mass fraction of the water constituent, in units of [g g^-1].
        dq : array-like
            Buffer which will store the output density of the water constituent,
            in the same units of the input air density.
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # run the stencil
        self._stencil_diagnosing_density(
            in_d=d,
            in_q=q,
            out_dq=dq,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    def get_mass_fraction_of_water_constituent_in_air(
        self: "WaterConstituent",
        d: typing.array_t,
        dq: typing.array_t,
        q: typing.array_t,
    ) -> None:
        """
        Diagnose the mass fraction of a water constituent.

        Parameters
        ----------
        d : array-like
            The air density.
        dq : array-like
            The density of the water constituent, in the same units of the input
            air density.
        q : array-like
            Buffer which will store the output mass fraction of the water
            constituent, in the same units of the input air density.
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # run the stencil
        self._stencil_diagnosing_mass_fraction(
            in_d=d,
            in_dq=dq,
            out_q=q,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    @stencil_definition(backend=("numpy", "cupy"), stencil="density")
    def _diagnose_density_numpy(
        self: "WaterConstituent",
        in_d: np.ndarray,
        in_q: np.ndarray,
        out_dq: np.ndarray,
        *,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_dq[i, j, k] = in_d[i, j, k] * in_q[i, j, k]
        if self._clipping:
            out_dq[i, j, k] = np.where(
                out_dq[i, j, k] > 0.0, out_dq[i, j, k], 0.0
            )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="density")
    def _diagnose_density_gt4py(
        in_d: gtscript.Field["dtype"],
        in_q: gtscript.Field["dtype"],
        out_dq: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import clipping, positive

        with computation(PARALLEL), interval(...):
            if __INLINED(clipping):  # compile-time if
                tmp_dq = in_d * in_q
                out_dq = positive(tmp_dq)
            else:
                out_dq = in_d * in_q

    @stencil_definition(backend=("numpy", "cupy"), stencil="mass_fraction")
    def _diagnose_mass_fraction_numpy(
        self: "WaterConstituent",
        in_d: np.ndarray,
        in_dq: np.ndarray,
        out_q: np.ndarray,
        *,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_q[i, j, k] = in_dq[i, j, k] / in_d[i, j, k]
        if self._clipping:
            out_q[i, j, k] = np.where(
                out_q[i, j, k] > 0.0, out_q[i, j, k], 0.0
            )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="mass_fraction")
    def _diagnose_mass_fraction_gt4py(
        in_d: gtscript.Field["dtype"],
        in_dq: gtscript.Field["dtype"],
        out_q: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import clipping, positive

        with computation(PARALLEL), interval(...):
            if __INLINED(clipping):  # compile-time if
                tmp_q = in_dq / in_d
                out_q = positive(tmp_q)
            else:
                out_q = in_dq / in_d
