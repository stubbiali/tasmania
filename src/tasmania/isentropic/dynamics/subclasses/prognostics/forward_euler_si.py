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

from sympl._core.time import Timer

from tasmania.isentropic.dynamics.diagnostics import IsentropicDiagnostics
from tasmania.isentropic.dynamics.horizontal_fluxes import IsentropicMinimalHorizontalFlux
from tasmania.isentropic.dynamics.prognostic import IsentropicPrognostic


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class ForwardEulerSI(IsentropicPrognostic):
    """The semi-implicit upwind scheme."""

    name = "forward_euler_si"

    def __init__(
        self,
        horizontal_flux_scheme,
        domain,
        moist,
        *,
        backend="numpy",
        backend_options=None,
        storage_shape=None,
        storage_options=None,
        **kwargs,
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
        self._pt = kwargs["pt"].to_units("Pa").values.item() if "pt" in kwargs else 0.0
        self._eps = kwargs.get("eps", 0.5)
        assert 0.0 <= self._eps <= 1.0, "The off-centering parameter should be between 0 and 1."

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

    def stage_call(self, stage, timestep, state, tendencies, out_state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self.horizontal_boundary.nb
        tendencies = tendencies or {}

        if self._stencil is None:
            # initialize the stencils
            self._stencils_initialize(tendencies)

        # set inputs for the first stencil
        dt = timestep.total_seconds()
        dx = self.grid.dx.to_units("m").values.item()
        dy = self.grid.dy.to_units("m").values.item()
        stencil_args = {
            "s_now": state["air_isentropic_density"],
            "s_int": state["air_isentropic_density"],
            "s_tnd": tendencies.get("air_isentropic_density", None),
            "s_new": out_state["air_isentropic_density"],
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
                    "qv_tnd": tendencies.get(mfwv, None),
                    "sqv_new": out_state["isentropic_density_of_water_vapor"],
                    "sqc_now": state["isentropic_density_of_cloud_liquid_water"],
                    "sqc_int": state["isentropic_density_of_cloud_liquid_water"],
                    "qc_tnd": tendencies.get(mfcw, None),
                    "sqc_new": out_state["isentropic_density_of_cloud_liquid_water"],
                    "sqr_now": state["isentropic_density_of_precipitation_water"],
                    "sqr_int": state["isentropic_density_of_precipitation_water"],
                    "qr_tnd": tendencies.get(mfpw, None),
                    "sqr_new": out_state["isentropic_density_of_precipitation_water"],
                }
            )

        # step the isentropic density and the water species
        Timer.start(label="stencil")
        self._stencil(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

        # apply the boundary conditions on the stepped isentropic density
        self.horizontal_boundary.enforce_field(
            out_state["air_isentropic_density"],
            "air_isentropic_density",
            "kg m^-2 K^-1",
            time=state["time"] + timestep,
        )

        # diagnose the Montgomery potential from the stepped isentropic density
        self._diagnostics.get_montgomery_potential(
            out_state["air_isentropic_density"], self._pt, self._mtg_new
        )

        # set inputs for the second stencil
        stencil_args = {
            "s_now": state["air_isentropic_density"],
            "s_int": state["air_isentropic_density"],
            "s_new": out_state["air_isentropic_density"],
            "u_int": state["x_velocity_at_u_locations"],
            "v_int": state["y_velocity_at_v_locations"],
            "mtg_now": state["montgomery_potential"],
            "mtg_new": self._mtg_new,
            "su_now": state["x_momentum_isentropic"],
            "su_int": state["x_momentum_isentropic"],
            "su_tnd": tendencies.get("x_momentum_isentropic"),
            "su_new": out_state["x_momentum_isentropic"],
            "sv_now": state["y_momentum_isentropic"],
            "sv_int": state["y_momentum_isentropic"],
            "sv_tnd": tendencies.get("y_momentum_isentropic"),
            "sv_new": out_state["y_momentum_isentropic"],
        }

        # step the momenta
        Timer.start(label="stencil")
        self._stencil_momentum(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            eps=self._eps,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

        # set time
        out_state["time"] = state["time"] + timestep

    def _stencils_allocate_temporaries(self) -> None:
        super()._stencils_allocate_temporaries()

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
                "flux_dry": self._hflux.get_subroutine_definition("flux_dry"),
                "flux_moist": self._hflux.get_subroutine_definition("flux_moist"),
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
        self._stencil = self.compile_stencil("step_forward_euler")
        self._stencil_momentum = self.compile_stencil("step_forward_euler_momentum")

        # allocate temporaries
        self._stencils_allocate_temporaries()
