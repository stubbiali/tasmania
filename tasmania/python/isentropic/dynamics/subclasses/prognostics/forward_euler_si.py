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
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics,
)
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.framework.register import register


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@register(name="forward_euler_si")
class ForwardEulerSI(IsentropicPrognostic):
    """The semi-implicit upwind scheme."""

    def __init__(
        self,
        horizontal_flux_scheme,
        domain,
        moist,
        backend,
        backend_options,
        storage_shape,
        storage_options,
        **kwargs
    ):
        # call parent's constructor
        super().__init__(
            IsentropicMinimalHorizontalFlux,
            horizontal_flux_scheme,
            domain,
            moist,
            backend,
            backend_options,
            storage_shape,
            storage_options,
        )

        # extract the upper boundary conditions on the pressure field and
        # the off-centering parameter for the semi-implicit integrator
        self._pt = (
            kwargs["pt"].to_units("Pa").values.item()
            if "pt" in kwargs
            else 0.0
        )
        self._eps = kwargs.get("eps", 0.5)
        assert (
            0.0 <= self._eps <= 1.0
        ), "The off-centering parameter should be between 0 and 1."

        # instantiate the component retrieving the diagnostic variables
        self._diagnostics = IsentropicDiagnostics(
            self.grid,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_shape=self._storage_shape,
            storage_options=self.storage_options,
        )

        # initialize the pointers to the stencil objects
        self._stencil = None
        self._stencil_momentum = None

    @property
    def stages(self):
        return 1

    @property
    def substep_fractions(self):
        return 1.0

    def stage_call(self, stage, timestep, state, tendencies=None):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self.horizontal_boundary.nb
        tendencies = tendencies or {}

        if self._stencil is None:
            # initialize the stencils
            self._stencils_initialize(tendencies)

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
        dt = timestep.total_seconds()
        dx = self._grid.dx.to_units("m").values.item()
        dy = self._grid.dy.to_units("m").values.item()
        stencil_args = {
            "s_now": state["air_isentropic_density"],
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
                    "sqv_now": state["isentropic_density_of_water_vapor"],
                    "sqv_int": state["isentropic_density_of_water_vapor"],
                    "qv_tnd": self._qv_tnd,
                    "sqv_new": self._sqv_new,
                    "sqc_now": state[
                        "isentropic_density_of_cloud_liquid_water"
                    ],
                    "sqc_int": state[
                        "isentropic_density_of_cloud_liquid_water"
                    ],
                    "qc_tnd": self._qc_tnd,
                    "sqc_new": self._sqc_new,
                    "sqr_now": state[
                        "isentropic_density_of_precipitation_water"
                    ],
                    "sqr_int": state[
                        "isentropic_density_of_precipitation_water"
                    ],
                    "qr_tnd": self._qr_tnd,
                    "sqr_new": self._sqr_new,
                }
            )

        # step the isentropic density and the water species
        self._stencil(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args
        )

        # apply the boundary conditions on the stepped isentropic density
        try:
            self.horizontal_boundary.dmn_enforce_field(
                self._s_new,
                "air_isentropic_density",
                "kg m^-2 K^-1",
                time=state["time"] + timestep,
            )
        except AttributeError:
            self.horizontal_boundary.enforce_field(
                self._s_new,
                "air_isentropic_density",
                "kg m^-2 K^-1",
                time=state["time"] + timestep,
            )

        # diagnose the Montgomery potential from the stepped isentropic density
        self._diagnostics.get_montgomery_potential(
            self._s_new, self._pt, self._mtg_new
        )

        # set inputs for the second stencil
        stencil_args = {
            "s_now": state["air_isentropic_density"],
            "s_int": state["air_isentropic_density"],
            "s_new": self._s_new,
            "u_int": state["x_velocity_at_u_locations"],
            "v_int": state["y_velocity_at_v_locations"],
            "mtg_now": state["montgomery_potential"],
            "mtg_new": self._mtg_new,
            "su_now": state["x_momentum_isentropic"],
            "su_int": state["x_momentum_isentropic"],
            "su_tnd": self._su_tnd,
            "su_new": self._su_new,
            "sv_now": state["y_momentum_isentropic"],
            "sv_int": state["y_momentum_isentropic"],
            "sv_tnd": self._sv_tnd,
            "sv_new": self._sv_new,
        }

        # step the momenta
        self._stencil_momentum(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            eps=self._eps,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args
        )

        # collect the outputs
        out_state = {
            "time": state["time"] + timestep,
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
        self._mtg_new = self.zeros(shape=self._storage_shape)

    def _stencils_initialize(self, tendencies):
        # set dtypes dictionary
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}

        # set external symbols
        externals = self._hflux.externals.copy()
        externals.update(
            {
                "extent": self._hflux.extent,
                "flux_dry": self._hflux.stencil_subroutine("flux_dry"),
                "flux_moist": self._hflux.stencil_subroutine("flux_moist"),
                "moist": self._moist,
                "s_tnd_on": "air_isentropic_density" in tendencies,
                "su_tnd_on": "x_momentum_isentropic" in tendencies,
                "sv_tnd_on": "y_momentum_isentropic" in tendencies,
                "qv_tnd_on": self._moist and mfwv in tendencies,
                "qc_tnd_on": self._moist and mfcw in tendencies,
                "qr_tnd_on": self._moist and mfpw in tendencies,
            }
        )
        self.backend_options.externals = externals

        # compile the stencils
        self._stencil = self.compile("step_forward_euler")
        self._stencil_momentum = self.compile("step_forward_euler_momentum")
