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
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
from property_cached import cached_property
import pytest

from tasmania.python.isentropic.physics.vertical_advection import (
    IsentropicVerticalAdvection,
    PrescribedSurfaceHeating,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.isentropic.test_isentropic_minimal_vertical_fluxes import (
    get_upwind_flux,
    get_centered_flux,
    get_third_order_upwind_flux,
    get_fifth_order_upwind_flux,
)
from tests.strategies import (
    st_isentropic_state_f,
    st_raw_field,
)
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicVerticalAdvectionTestSuite(TendencyComponentTestSuite):
    def __init__(
        self,
        domain_suite,
        flux_scheme,
        moist,
        toaptoil,
    ):
        self.flux_scheme = flux_scheme
        self.moist = moist
        self.toaptoil = toaptoil
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return IsentropicVerticalAdvection(
            self.ds.domain,
            self.ds.grid_type,
            self.flux_scheme,
            self.moist,
            tendency_of_air_potential_temperature_on_interface_levels=self.toaptoil,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        state = self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=True,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        field = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz),
                -1e4,
                1e4,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="field",
        )
        state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
            field,
            self.ds.grid,
            "K s^-1",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        field = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz + 1),
                -1e4,
                1e4,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="field",
        )
        state[
            "tendency_of_air_potential_temperature_on_interface_levels"
        ] = get_dataarray_3d(
            field,
            self.ds.grid,
            "K s^-1",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        )
        return state

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz

        if self.toaptoil:
            name = "tendency_of_air_potential_temperature_on_interface_levels"
            w = raw_state_np[name]
            w_hl = w
        else:
            name = "tendency_of_air_potential_temperature"
            w = raw_state_np[name]
            w_hl = np.zeros(
                self.component.get_field_storage_shape(
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ),
                dtype=self.ds.so.dtype,
            )
            w_hl[:nx, :ny, 1:nz] = 0.5 * (
                w[:nx, :ny, : nz - 1] + w[:nx, :ny, 1:nz]
            )

        s = raw_state_np["air_isentropic_density"]
        su = raw_state_np["x_momentum_isentropic"]
        sv = raw_state_np["y_momentum_isentropic"]
        if self.moist:
            qv = raw_state_np[mfwv]
            sqv = s * qv
            qc = raw_state_np[mfcw]
            sqc = s * qc
            qr = raw_state_np[mfpw]
            sqr = s * qr

        nb = self.flux_properties[self.flux_scheme]["nb"]
        get_flux = self.flux_properties[self.flux_scheme]["get_flux"]
        # set_lower_layers = self.flux_properties[self.flux_scheme]["set_lower_layers"]
        up = slice(nb, nz - nb)
        down = slice(nb + 1, nz - nb + 1)
        dz = self.ds.grid.dz.to_units("K").values.item()

        tendencies, diagnostics = {}, {}

        flux = get_flux(w_hl, s)
        s_out = np.zeros_like(s)
        s_out[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
        # set_lower_layers(nb, dz, w, s, out, staggering=toaptoil)
        tendencies["air_isentropic_density"] = s_out

        flux = get_flux(w_hl, su)
        su_out = np.zeros_like(su)
        su_out[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
        # set_lower_layers(nb, dz, w, su, out, staggering=toaptoil)
        tendencies["x_momentum_isentropic"] = su_out

        flux = get_flux(w_hl, sv)
        sv_out = np.zeros_like(sv)
        sv_out[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
        # set_lower_layers(nb, dz, w, sv, out, staggering=toaptoil)
        tendencies["y_momentum_isentropic"] = sv_out

        if self.moist:
            flux = get_flux(w_hl, sqv)
            out_qv = np.zeros_like(qv)
            out_qv[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
            # set_lower_layers(nb, dz, w, sqv, out, staggering=toaptoil)
            out_qv /= s
            tendencies[mfwv] = out_qv

            flux = get_flux(w_hl, sqc)
            out_qc = np.zeros_like(qc)
            out_qc[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
            # set_lower_layers(nb, dz, w, sqc, out, staggering=toaptoil)
            out_qc /= s
            tendencies[mfcw] = out_qc

            flux = get_flux(w_hl, sqr)
            out_qr = np.zeros_like(qr)
            out_qr[:, :, up] = -(flux[:, :, up] - flux[:, :, down]) / dz
            # set_lower_layers(nb, dz, w, sqr, out, staggering=toaptoil)
            out_qr /= s
            tendencies[mfpw] = out_qr

        return tendencies, diagnostics

    @cached_property
    def flux_properties(self):
        return {
            "upwind": {
                "nb": 1,
                "get_flux": get_upwind_flux,
                "set_lower_layers": self.set_lower_layers_first_order,
            },
            "centered": {
                "nb": 1,
                "get_flux": get_centered_flux,
                "set_lower_layers": self.set_lower_layers_second_order,
            },
            "third_order_upwind": {
                "nb": 2,
                "get_flux": get_third_order_upwind_flux,
                "set_lower_layers": self.set_lower_layers_second_order,
            },
            # "fifth_order_upwind": {
            #     "nb": 3,
            #     "get_flux": get_fifth_order_upwind_flux,
            #     "set_lower_layers": self.set_lower_layers_second_order,
            # },
        }

    @staticmethod
    def set_lower_layers_first_order(nb, dz, w, phi, out, staggering=False):
        wm = np.zeros_like(w)
        wm[:, :, :-1] = (
            0.5 * (w[:, :, :-1] + w[:, :, 1:]) if staggering else w[:, :, :-1]
        )
        out[:, :, -nb - 1 : -1] = (
            1
            / dz
            * (
                wm[:, :, -nb - 2 : -2] * phi[:, :, -nb - 2 : -2]
                - wm[:, :, -nb - 1 : -1] * phi[:, :, -nb - 1 : -1]
            )
        )

    @staticmethod
    def set_lower_layers_second_order(nb, dz, w, phi, out, staggering=False):
        wm = np.zeros_like(w)
        wm[:, :, :-1] = (
            0.5 * (w[:, :, :-1] + w[:, :, 1:]) if staggering else w[:, :, :-1]
        )
        out[:, :, -nb - 1 : -1] = (
            0.5
            * (
                -3.0 * wm[:, :, -nb - 1 : -1] * phi[:, :, -nb - 1 : -1]
                + 4.0 * wm[:, :, -nb - 2 : -2] * phi[:, :, -nb - 2 : -2]
                - wm[:, :, -nb - 3 : -3] * phi[:, :, -nb - 3 : -3]
            )
            / dz
        )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "flux_scheme", ("upwind", "centered", "third_order_upwind")
)
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, flux_scheme, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype, zaxis_length=(5, 50))

    # ========================================
    # test bed
    # ========================================
    ts = IsentropicVerticalAdvectionTestSuite(
        ds, flux_scheme, moist=False, toaptoil=False
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        ds, flux_scheme, moist=False, toaptoil=True
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        ds, flux_scheme, moist=True, toaptoil=False
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        ds, flux_scheme, moist=True, toaptoil=True
    )
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
    # test("upwind", "numpy", float, None)
