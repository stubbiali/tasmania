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
import numpy as np
from typing import Dict, TYPE_CHECKING

from sympl._core.time import Timer

from gt4py import gtscript

from tasmania.python.framework.tag import stencil_definition
from tasmania.python.physics.turbulence import Smagorinsky2d

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict


class IsentropicSmagorinsky(Smagorinsky2d):
    """
    Implementation of the Smagorinsky turbulence model for the
    isentropic model. The conservative form of the governing
    equations is used.
    The class is instantiated over the *numerical* grid of the
    underlying domain.
    """

    @property
    def input_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
        }

    @property
    def tendency_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        return {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }

    @property
    def diagnostic_properties(self) -> "PropertyDict":
        return {}

    def array_call(
        self,
        state: "NDArrayLikeDict",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        Timer.start(label="stencil")
        self._stencil(
            in_s=state["air_isentropic_density"],
            in_su=state["x_momentum_isentropic"],
            in_sv=state["y_momentum_isentropic"],
            out_su_tnd=out_tendencies["x_momentum_isentropic"],
            out_sv_tnd=out_tendencies["y_momentum_isentropic"],
            dx=dx,
            dy=dy,
            cs=self._cs,
            ow_out_su_tnd=overwrite_tendencies["x_momentum_isentropic"],
            ow_out_sv_tnd=overwrite_tendencies["y_momentum_isentropic"],
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="smagorinsky")
    def _stencil_numpy(
        in_s: np.ndarray,
        in_su: np.ndarray,
        in_sv: np.ndarray,
        out_su_tnd: np.ndarray,
        out_sv_tnd: np.ndarray,
        *,
        dx: float,
        dy: float,
        cs: float,
        ow_out_su_tnd: bool,
        ow_out_sv_tnd: bool,
        origin: "TripletInt",
        domain: "TripletInt",
    ) -> None:
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        u = in_su / in_s
        v = in_sv / in_s
        u_tnd, v_tnd = core(u, v, dx, dy, cs, ib, ie, jb, je, k)
        tmp_out_su_tnd = in_s[ib:ie, jb:je, k] * u_tnd
        tmp_out_sv_tnd = in_s[ib:ie, jb:je, k] * v_tnd
        set_output(out_su_tnd[ib:ie, jb:je, k], tmp_out_su_tnd, ow_out_su_tnd)
        set_output(out_sv_tnd[ib:ie, jb:je, k], tmp_out_sv_tnd, ow_out_sv_tnd)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smagorinsky")
    def _stencil_gt4py(
        in_s: gtscript.Field["dtype"],
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        out_su_tnd: gtscript.Field["dtype"],
        out_sv_tnd: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        cs: float,
        ow_out_su_tnd: bool,
        ow_out_sv_tnd: bool,
    ) -> None:
        from __externals__ import core, set_output

        with computation(PARALLEL), interval(...):
            u = in_su / in_s
            v = in_sv / in_s
            u_tnd, v_tnd = core(u, v, dx, dy, cs)
            tmp_out_su_tnd = in_s * u_tnd
            tmp_out_sv_tnd = in_s * v_tnd
            out_su_tnd = set_output(out_su_tnd, tmp_out_su_tnd, ow_out_su_tnd)
            out_sv_tnd = set_output(out_sv_tnd, tmp_out_sv_tnd, ow_out_sv_tnd)
