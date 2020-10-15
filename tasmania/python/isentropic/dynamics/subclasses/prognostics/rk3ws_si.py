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
from gt4py import gtscript

from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics,
)
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.isentropic.dynamics.subclasses.prognostics.utils import (
    step_forward_euler_gt,
    step_forward_euler_momentum_gt,
    step_forward_euler_momentum_numpy,
    step_forward_euler_numpy,
)
from tasmania.python.utils.framework_utils import register
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@register(name="rk3ws_si")
class RK3WSSI(IsentropicPrognostic):
    """ The semi-implicit three-stages Runge-Kutta scheme. """

    def __init__(
        self,
        horizontal_flux_scheme,
        grid,
        hb,
        moist,
        backend,
        backend_opts,
        dtype,
        build_info,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
        managed_memory,
        **kwargs
    ):
        # call parent's constructor
        super().__init__(
            IsentropicMinimalHorizontalFlux,
            horizontal_flux_scheme,
            grid,
            hb,
            moist,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        )

        # extract the upper boundary conditions on the pressure field and
        # the off-centering parameter for the semi-implicit integrator
        self._pt = (
            kwargs["pt"].to_units("Pa").data.item() if "pt" in kwargs else 0.0
        )
        self._eps = kwargs.get("eps", 0.5)
        assert (
            0.0 <= self._eps <= 1.0
        ), "The off-centering parameter should be between 0 and 1."

        # instantiate the component retrieving the diagnostic variables
        self._diagnostics = IsentropicDiagnostics(
            grid,
            backend=backend,
            backend_opts=backend_opts,
            dtype=dtype,
            build_info=build_info,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=False,
            storage_shape=storage_shape,
            managed_memory=managed_memory,
        )

        # initialize the pointers to the stencils
        self._stencil = None
        self._stencil_momentum = None

        # initialize the pointers to the solution at the current timestep
        self._s_now = None
        self._mtg_now = None
        self._su_now = None
        self._sv_now = None
        if moist:
            self._sqv_now = None
            self._sqc_now = None
            self._sqr_now = None

    @property
    def stages(self):
        return 3

    @property
    def substep_fractions(self):
        return 1.0 / 3.0, 0.5, 1.0

    def stage_call(self, stage, timestep, state, tendencies=None):
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        nb = self._hb.nb
        tendencies = {} if tendencies is None else tendencies

        if self._stencil is None:
            # initialize the stencils
            self._stencils_initialize(tendencies)

        # set the correct timestep
        if stage == 0:
            dtr = timestep / 3.0
            dt = timestep / 3.0
        elif stage == 1:
            dtr = timestep / 6.0
            dt = 0.5 * timestep
        else:
            dtr = 0.5 * timestep
            dt = timestep

        # keep track of the current state
        if stage == 0:
            self._s_now = state["air_isentropic_density"]
            self._mtg_now = state["montgomery_potential"]
            self._su_now = state["x_momentum_isentropic"]
            self._sv_now = state["y_momentum_isentropic"]
            if self._moist:
                self._sqv_now = state["isentropic_density_of_water_vapor"]
                self._sqc_now = state[
                    "isentropic_density_of_cloud_liquid_water"
                ]
                self._sqr_now = state[
                    "isentropic_density_of_precipitation_water"
                ]

        # grab the tendencies
        if "air_isentropic_density" in tendencies:
            self._s_tnd = tendencies["air_isentropic_density"]
        if "x_momentum_isentropic" in tendencies:
            self._su_tnd = tendencies["x_momentum_isentropic"]
        if "y_momentum_isentropic" in tendencies:
            self._sv_tnd = tendencies["y_momentum_isentropic"]
        if self._moist:
            if mfwv in tendencies:
                self._qv_tnd = tendencies[mfwv]
            if mfcw in tendencies:
                self._qc_tnd = tendencies[mfcw]
            if mfpw in tendencies:
                self._qr_tnd = tendencies[mfpw]

        # set inputs for the first stencil
        dt = dt.total_seconds()
        dx = self._grid.dx.to_units("m").values.item()
        dy = self._grid.dy.to_units("m").values.item()
        stencil_args = {
            "s_now": self._s_now,
            "s_int": state["air_isentropic_density"],
            "s_tnd": self._s_tnd,
            "s_new": self._s_new,
            "u_int": state["x_velocity_at_u_locations"],
            "v_int": state["y_velocity_at_v_locations"],
            "su_int": state["x_momentum_isentropic"],
            "sv_int": state["y_momentum_isentropic"],
        }
        if self._moist:
            stencil_args.update(
                {
                    "sqv_now": self._sqv_now,
                    "sqv_int": state["isentropic_density_of_water_vapor"],
                    "qv_tnd": self._qv_tnd,
                    "sqv_new": self._sqv_new,
                    "sqc_now": self._sqc_now,
                    "sqc_int": state[
                        "isentropic_density_of_cloud_liquid_water"
                    ],
                    "qc_tnd": self._qc_tnd,
                    "sqc_new": self._sqc_new,
                    "sqr_now": self._sqr_now,
                    "sqr_int": state[
                        "isentropic_density_of_precipitation_water"
                    ],
                    "qr_tnd": self._qr_tnd,
                    "sqr_new": self._sqr_new,
                }
            )
        if not is_gt(self._backend):
            stencil_args["fluxer"] = self._hflux

        # step the isentropic density and the water species
        self._stencil(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
            validate_args=False
        )

        # apply the boundary conditions on the stepped isentropic density
        try:
            self._hb.dmn_enforce_field(
                self._s_new,
                "air_isentropic_density",
                "kg m^-2 K^-1",
                time=state["time"] + dtr,
            )
        except AttributeError:
            self._hb.enforce_field(
                self._s_new,
                "air_isentropic_density",
                "kg m^-2 K^-1",
                time=state["time"] + dtr,
            )

        # diagnose the Montgomery potential from the stepped isentropic density
        self._diagnostics.get_montgomery_potential(
            self._s_new, self._pt, self._mtg_new
        )

        # set inputs for the second stencil
        stencil_args = {
            "s_now": self._s_now,
            "s_int": state["air_isentropic_density"],
            "s_new": self._s_new,
            "u_int": state["x_velocity_at_u_locations"],
            "v_int": state["y_velocity_at_v_locations"],
            "mtg_now": self._mtg_now,
            "mtg_new": self._mtg_new,
            "su_now": self._su_now,
            "su_int": state["x_momentum_isentropic"],
            "su_tnd": self._su_tnd,
            "su_new": self._su_new,
            "sv_now": self._sv_now,
            "sv_int": state["y_momentum_isentropic"],
            "sv_tnd": self._sv_tnd,
            "sv_new": self._sv_new,
        }
        if not is_gt(self._backend):
            stencil_args["fluxer"] = self._hflux

        # step the momenta
        self._stencil_momentum(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            eps=self._eps,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
            validate_args=False
        )

        # collect the outputs
        out_state = {
            "time": state["time"] + dtr,
            "air_isentropic_density": self._s_new,
            "x_momentum_isentropic": self._su_new,
            "y_momentum_isentropic": self._sv_new,
        }
        if self._moist:
            out_state.update(
                {
                    "isentropic_density_of_water_vapor": self._sqv_new,
                    "isentropic_density_of_cloud_liquid_water": self._sqc_new,
                    "isentropic_density_of_precipitation_water": self._sqr_new,
                }
            )

        return out_state

    def _stencils_allocate_outputs(self):
        super()._stencils_allocate_outputs()

        # allocate the storage which will collect the Montgomery potential
        # retrieved from the updated isentropic density
        storage_shape = self._storage_shape
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        managed_memory = self._managed_memory
        self._mtg_new = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

    def _stencils_initialize(self, tendencies):
        if is_gt(self._backend):
            # set external symbols for the first stencil
            externals = {
                "fluxer": self._hflux.call,
                "moist": self._moist,
                "s_tnd_on": "air_isentropic_density" in tendencies,
                "su_tnd_on": False,
                "sv_tnd_on": False,
                "qv_tnd_on": self._moist and mfwv in tendencies,
                "qc_tnd_on": self._moist and mfcw in tendencies,
                "qr_tnd_on": self._moist and mfpw in tendencies,
            }

            # compile the first stencil
            self._stencil = gtscript.stencil(
                definition=step_forward_euler_gt,
                # name=self.__class__.__name__ + "_stencil",
                backend=get_gt_backend(self._backend),
                build_info=self._build_info,
                dtypes={"dtype": self._dtype},
                externals=externals,
                rebuild=self._rebuild,
                **self._backend_opts
            )

            # set external symbols for the second stencil
            externals = {
                "fluxer": self._hflux.call,
                "moist": False,
                "s_tnd_on": False,
                "su_tnd_on": "x_momentum_isentropic" in tendencies,
                "sv_tnd_on": "y_momentum_isentropic" in tendencies,
                "qv_tnd_on": False,
                "qc_tnd_on": False,
                "qr_tnd_on": False,
            }

            # compile the second stencil
            self._stencil_momentum = gtscript.stencil(
                definition=step_forward_euler_momentum_gt,
                # name=self.__class__.__name__ + "_stencil_momentum",
                backend=get_gt_backend(self._backend),
                build_info=self._build_info,
                dtypes={"dtype": self._dtype},
                externals=externals,
                rebuild=self._rebuild,
                **self._backend_opts
            )
        else:
            self._stencil = step_forward_euler_numpy
            self._stencil_momentum = step_forward_euler_momentum_numpy
