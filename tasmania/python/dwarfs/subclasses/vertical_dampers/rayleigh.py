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
from sympl import DataArray

from gt4py import gtscript

from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.register import register
from tasmania.python.framework.tag import stencil_definition


@register(name="rayleigh")
class Rayleigh(VerticalDamping):
    """ The Rayleigh absorber. """

    def __init__(
        self,
        grid,
        damp_depth=15,
        damp_coeff_max=0.0002,
        time_units="s",
        backend="numpy",
        backend_options=None,
        storage_shape=None,
        storage_options=None,
    ):
        super().__init__(
            grid,
            damp_depth,
            damp_coeff_max,
            time_units,
            backend,
            backend_options,
            storage_shape,
            storage_options,
        )

    def __call__(self, dt, field_now, field_new, field_ref, field_out):
        # shortcuts
        ni, nj, nk = self._shape
        dnk = self._damp_depth

        # convert the timestep to seconds
        dt_da = DataArray(dt.total_seconds(), attrs={"units": "s"})
        dt_raw = dt_da.to_units(self._tunits).values.item()

        # run the stencil
        self._stencil(
            in_phi_now=field_now,
            in_phi_new=field_new,
            in_phi_ref=field_ref,
            in_rmat=self._rmat,
            out_phi=field_out,
            dt=dt_raw,
            origin=(0, 0, 0),
            domain=(ni, nj, nk),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

        # if nk > dnk:
        #     # set the lowermost layers, outside of the damping region
        #     if not self._gt_powered:
        #         field_out[:, :, dnk:nk] = field_new[:, :, dnk:nk]

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="damping")
    def _damping_numpy(
        in_phi_now,
        in_phi_new,
        in_phi_ref,
        in_rmat,
        out_phi,
        *,
        dt,
        origin,
        domain,
        **kwargs
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = in_phi_new[i, j, k] - dt * in_rmat[i, j, k] * (
            in_phi_now[i, j, k] - in_phi_ref[i, j, k]
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="damping")
    def _damping_gt4py(
        in_phi_now: gtscript.Field["dtype"],
        in_phi_new: gtscript.Field["dtype"],
        in_phi_ref: gtscript.Field["dtype"],
        in_rmat: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            if in_rmat > 0.0:
                out_phi = in_phi_new - dt * in_rmat * (in_phi_now - in_phi_ref)
            else:
                out_phi = in_phi_new
