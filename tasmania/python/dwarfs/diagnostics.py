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

from gridtools import gtscript
# from gridtools.__gtscript__ import computation, interval, PARALLEL

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


class HorizontalVelocity:
    """
    This class diagnoses the horizontal momenta (respectively, velocity
    components) with the help of the air density and the horizontal
    velocity components (resp., momenta).
    """

    def __init__(
        self,
        grid,
        staggering=True,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        exec_info=None,
        rebuild=False
    ):
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        staggering : `bool`, optional
            :obj:`True` if the velocity components should be computed
            on the staggered grid, :obj:`False` to collocate the velocity
            components in the mass points.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        """
        # store input arguments needed at run-time
        self._grid = grid
        self._staggering = staggering
        self._exec_info = exec_info

        # initialize the underlying stencils
        self._stencil_diagnosing_momenta = gtscript.stencil(
            definition=self._stencil_diagnosing_momenta_defs,
            backend=backend,
            build_info=build_info,
            externals={"staggering": staggering},
            rebuild=rebuild,
            **(backend_opts or {})
        )
        self._stencil_diagnosing_velocity_x = gtscript.stencil(
            definition=self._stencil_diagnosing_velocity_x_defs,
            backend=backend,
            build_info=build_info,
            externals={"staggering": staggering},
            rebuild=rebuild,
            **(backend_opts or {})
        )
        self._stencil_diagnosing_velocity_y = gtscript.stencil(
            definition=self._stencil_diagnosing_velocity_y_defs,
            backend=backend,
            build_info=build_info,
            externals={"staggering": staggering},
            rebuild=rebuild,
            **(backend_opts or {})
        )

    def get_momenta(self, d, u, v, du, dv):
        """
        Diagnose the horizontal momenta.

        Parameters
        ----------
        d : gridtools.storage.Storage
            The air density.
        u : gridtools.storage.Storage
            The x-velocity field.
        v : gridtools.storage.Storage
            The y-velocity field.
        du : gridtools.storage.Storage
            The buffer where the x-momentum will be written.
        dv : gridtools.storage.Storage
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
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

    def get_velocity_components(self, d, du, dv, u, v):
        """
        Diagnose the horizontal velocity components.

        Parameters
        ----------
        d : gridtools.storage.Storage
            The air density.
        du : gridtools.storage.Storage
            The x-momentum.
        dv : gridtools.storage.Storage
            The y-momentum.
        u : gridtools.storage.Storage
            The buffer where the x-velocity will be written.
        v : gridtools.storage.Storage
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
            origin={"_all_": (dn, 0, 0)},
            domain=(nx - dn, ny, nz),
            exec_info=self._exec_info,
        )
        self._stencil_diagnosing_velocity_y(
            in_d=d,
            in_dv=dv,
            out_v=v,
            origin={"_all_": (0, dn, 0)},
            domain=(nx, ny - dn, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_diagnosing_momenta_defs(
        in_d: gtscript.Field[np.float64],
        in_u: gtscript.Field[np.float64],
        in_v: gtscript.Field[np.float64],
        out_du: gtscript.Field[np.float64],
        out_dv: gtscript.Field[np.float64],
    ):
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if staggering:  # compile-time if
                out_du = 0.5 * in_d[0, 0, 0] * (in_u[0, 0, 0] + in_u[1, 0, 0])
                out_dv = 0.5 * in_d[0, 0, 0] * (in_v[0, 0, 0] + in_v[0, 1, 0])
            else:
                out_du = in_d[0, 0, 0] * in_u[0, 0, 0]
                out_dv = in_d[0, 0, 0] * in_v[0, 0, 0]

    @staticmethod
    def _stencil_diagnosing_velocity_x_defs(
        in_d: gtscript.Field[np.float64],
        in_du: gtscript.Field[np.float64],
        out_u: gtscript.Field[np.float64],
    ):
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if staggering:  # compile-time if
                out_u = (in_du[-1, 0, 0] + in_du[0, 0, 0]) / (in_d[-1, 0, 0] + in_d[0, 0, 0])
            else:
                out_u = in_du[0, 0, 0] / in_d[0, 0, 0]

    @staticmethod
    def _stencil_diagnosing_velocity_y_defs(
        in_d: gtscript.Field[np.float64],
        in_dv: gtscript.Field[np.float64],
        out_v: gtscript.Field[np.float64],
    ):
        from __externals__ import staggering

        with computation(PARALLEL), interval(...):
            if staggering:  # compile-time if
                out_v = (in_dv[0, -1, 0] + in_dv[0, 0, 0]) / (in_d[0, -1, 0] + in_d[0, 0, 0])
            else:
                out_v = in_dv[0, 0, 0] / in_d[0, 0, 0]


class WaterConstituent:
    """
    This class diagnoses the density (respectively, mass fraction) of any water
    constituent with the help of the air density and the mass fraction (resp.,
    the density) of that water constituent.
    """

    def __init__(
        self,
        grid,
        clipping=False,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        exec_info=None,
        rebuild=False
    ):
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        clipping : `bool`, optional
            :obj:`True` to clip the negative values of the output fields,
            :obj:`False` otherwise. Defaults to :obj:`False`.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        """
        # store input arguments needed at run-time
        self._grid = grid
        self._exec_info = exec_info

        # initialize the underlying stencils
        self._stencil_diagnosing_density = gtscript.stencil(
            definition=self._stencil_diagnosing_density_defs,
            backend=backend,
            build_info=build_info,
            externals={"clipping": clipping},
            rebuild=rebuild,
            **(backend_opts or {})
        )
        self._stencil_diagnosing_mass_fraction = gtscript.stencil(
            definition=self._stencil_diagnosing_mass_fraction_defs,
            backend=backend,
            build_info=build_info,
            externals={"clipping": clipping},
            rebuild=rebuild,
            **(backend_opts or {})
        )

    def get_density_of_water_constituent(self, d, q, dq):
        """
        Diagnose the density of a water constituent.

        Parameters
        ----------
        d : gridtools.storage.Storage
            The air density.
        q : gridtools.storage.Storage
            The mass fraction of the water constituent, in units of [g g^-1].
        dq : gridtools.storage.Storage
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
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

    def get_mass_fraction_of_water_constituent_in_air(self, d, dq, q):
        """
        Diagnose the mass fraction of a water constituent.

        Parameters
        ----------
        d : gridtools.storage.Storage
            The air density.
        dq : gridtools.storage.Storage
            The density of the water constituent, in the same units of the input
            air density.
        q : gridtools.storage.Storage
            Buffer which will store the output mass fraction of the water constituent,
            in the same units of the input air density.
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # run the stencil
        self._stencil_diagnosing_mass_fraction(
            in_d=d,
            in_dq=dq,
            out_q=q,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_diagnosing_density_defs(
        in_d: gtscript.Field[np.float64],
        in_q: gtscript.Field[np.float64],
        out_dq: gtscript.Field[np.float64],
    ):
        from __externals__ import clipping

        with computation(PARALLEL), interval(...):
            if clipping:  # compile-time if
                tmp_dq = in_d[0, 0, 0] * in_q[0, 0, 0]
                out_dq = (tmp_dq[0, 0, 0] > 0.0) * tmp_dq[0, 0, 0]
            else:
                out_dq = in_d[0, 0, 0] * in_q[0, 0, 0]

    @staticmethod
    def _stencil_diagnosing_mass_fraction_defs(
        in_d: gtscript.Field[np.float64],
        in_dq: gtscript.Field[np.float64],
        out_q: gtscript.Field[np.float64],
    ):
        from __externals__ import clipping

        with computation(PARALLEL), interval(...):
            if clipping:  # compile-time if
                tmp_q = in_dq[0, 0, 0] / in_d[0, 0, 0]
                out_q = (tmp_q[0, 0, 0] > 0.0) * tmp_q[0, 0, 0]
            else:
                out_q = in_dq[0, 0, 0] / in_d[0, 0, 0]
