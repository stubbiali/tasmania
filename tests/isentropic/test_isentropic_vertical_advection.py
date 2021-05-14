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
from tests.suites import DomainSuite, TendencyComponentTestSuite
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicVerticalAdvectionTestSuite(TendencyComponentTestSuite):
    def __init__(
        self,
        hyp_data,
        domain_suite,
        flux_scheme,
        moist,
        toaptoil,
        *,
        storage_shape
    ):
        self.flux_scheme = flux_scheme
        self.moist = moist
        self.toaptoil = toaptoil
        self.storage_shape = storage_shape
        super().__init__(hyp_data, domain_suite)

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
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        state = self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=True,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )
        field = self.hyp_data.draw(
            st_raw_field(
                self.storage_shape,
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
            grid_shape=(self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz),
            set_coordinates=False,
        )
        state[
            "tendency_of_air_potential_temperature_on_interface_levels"
        ] = get_dataarray_3d(
            field,
            self.ds.grid,
            "K s^-1",
            grid_shape=(self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz + 1),
            set_coordinates=False,
        )
        return state

    def get_tendencies_and_diagnostics(self, raw_state_np):
        if self.toaptoil:
            name = "tendency_of_air_potential_temperature_on_interface_levels"
            w = raw_state_np[name]
            w_hl = w
        else:
            name = "tendency_of_air_potential_temperature"
            w = raw_state_np[name]
            w_hl = np.zeros_like(w)
            w_hl[:, :, 1:-1] = 0.5 * (w[:, :, :-2] + w[:, :, 1:-1])

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
        nz = self.ds.grid.nz
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
    storage_shape = (ds.grid.nx + 1, ds.grid.ny + 1, ds.grid.nz + 1)

    # ========================================
    # test bed
    # ========================================
    ts = IsentropicVerticalAdvectionTestSuite(
        data,
        ds,
        flux_scheme,
        moist=False,
        toaptoil=False,
        storage_shape=storage_shape,
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        data,
        ds,
        flux_scheme,
        moist=False,
        toaptoil=True,
        storage_shape=storage_shape,
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        data,
        ds,
        flux_scheme,
        moist=True,
        toaptoil=False,
        storage_shape=storage_shape,
    )
    ts.run()

    ts = IsentropicVerticalAdvectionTestSuite(
        data,
        ds,
        flux_scheme,
        moist=True,
        toaptoil=True,
        storage_shape=storage_shape,
    )
    ts.run()


# @hyp_settings
# @given(data=hyp_st.data())
# @pytest.mark.parametrize("backend", conf.backend)
# @pytest.mark.parametrize("dtype", conf.dtype)
# def _test_prescribed_surface_heating(data):
#     # ========================================
#     # random data generation
#     # ========================================
#     domain = data.draw(st_domain(), label="domain")
#     grid = domain.numerical_grid
#
#     time = data.draw(
#         hyp_st.datetimes(
#             min_value=datetime(1992, 2, 20), max_value=datetime(2010, 7, 21)
#         )
#     )
#     field = data.draw(
#         st_arrays(
#             grid.x.dtype,
#             (grid.nx, grid.ny, grid.nz + 1),
#             elements=st_floats(min_value=1, max_value=1e4),
#             fill=hyp_st.nothing(),
#         )
#     )
#     state = {
#         "time": time,
#         "air_pressure": get_dataarray_3d(field[:, :, : grid.nz], grid, "Pa"),
#         "air_pressure_on_interface_levels": get_dataarray_3d(field, grid, "Pa"),
#         "height_on_interface_levels": get_dataarray_3d(field, grid, "m"),
#     }
#
#     f0d_sw = data.draw(st_floats(min_value=0, max_value=100))
#     f0d_fw = data.draw(st_floats(min_value=0, max_value=100))
#     f0n_sw = data.draw(st_floats(min_value=0, max_value=100))
#     f0n_fw = data.draw(st_floats(min_value=0, max_value=100))
#     w_sw = data.draw(st_floats(min_value=0.1, max_value=100))
#     w_fw = data.draw(st_floats(min_value=0.1, max_value=100))
#     ad = data.draw(st_floats(min_value=0, max_value=100))
#     an = data.draw(st_floats(min_value=0, max_value=100))
#     cl = data.draw(st_floats(min_value=0, max_value=100))
#     t0 = data.draw(
#         hyp_st.datetimes(
#             min_value=datetime(1992, 2, 20), max_value=datetime(2010, 7, 21)
#         )
#     )
#
#     backend = data.draw(st_one_of(conf.backend), label="backend")
#
#     # ========================================
#     # test bed
#     # ========================================
#     f0d_sw_da = DataArray(f0d_sw, attrs={"units": "W m^-2"})
#     f0d_fw_da = DataArray(f0d_fw, attrs={"units": "W m^-2"})
#     f0n_sw_da = DataArray(f0n_sw, attrs={"units": "W m^-2"})
#     f0n_fw_da = DataArray(f0n_fw, attrs={"units": "W m^-2"})
#     w_sw_da = DataArray(w_sw, attrs={"units": "hr^-1"})
#     w_fw_da = DataArray(w_fw, attrs={"units": "hr^-1"})
#     ad_da = DataArray(ad, attrs={"units": "m^-1"})
#     an_da = DataArray(an, attrs={"units": "m^-1"})
#     cl_da = DataArray(cl, attrs={"units": "m"})
#
#     rd = 287.0
#     cp = 1004.0
#
#     nx, ny, nz = grid.nx, grid.ny, grid.nz
#     dtype = grid.x.dtype
#
#     x1d = grid.x.to_units("m").values
#     y1d = grid.y.to_units("m").values
#     x = np.tile(x1d[:, np.newaxis, np.newaxis], (1, ny, 1))
#     y = np.tile(y1d[np.newaxis, :, np.newaxis], (nx, 1, 1))
#
#     dt = (state["time"] - t0).total_seconds() / 3600.0
#
#     t = state["time"].hour * 3600 + state["time"].minute * 60 + state["time"].second
#     t_sw = 2 * np.pi / w_sw * 3600
#     isday = int(t / t_sw) % 2 == 0
#     f0_sw = f0d_sw if isday else f0n_sw
#     f0_fw = f0d_fw if isday else f0n_fw
#     a = ad if isday else an
#
#     #
#     # tendency_of_air_potential_temperature_on_interface_levels=False
#     # air_pressure_on_interface_levels=False
#     #
#     if state["time"] > t0:
#         theta = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
#         p = state["air_pressure"].values
#         z = 0.5 * (
#             state["height_on_interface_levels"].values[:, :, :-1]
#             + state["height_on_interface_levels"].values[:, :, 1:]
#         )
#         h = state["height_on_interface_levels"].values[:, :, -1:]
#
#         out = (
#             theta
#             * rd
#             * a
#             / (p * cp)
#             * np.exp(-a * (z - h))
#             * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
#             * (x ** 2 + y ** 2 < cl ** 2)
#         )
#     else:
#         out = np.zeros((nx, ny, nz), dtype=dtype)
#
#     # tendency_of_air_potential_temperature_in_diagnostics=False
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=False,
#         tendency_of_air_potential_temperature_on_interface_levels=False,
#         air_pressure_on_interface_levels=False,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "air_potential_temperature" in psf.tendency_properties
#     tendencies, diagnostics = psf(state)
#     assert "air_potential_temperature" in tendencies
#     compare_dataarrays(
#         out, tendencies["air_potential_temperature"], compare_coordinate_values=False
#     )
#     assert diagnostics == {}
#
#     # tendency_of_air_potential_temperature_in_diagnostics=True
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=True,
#         tendency_of_air_potential_temperature_on_interface_levels=False,
#         air_pressure_on_interface_levels=False,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
#     tendencies, diagnostics = psf(state)
#     assert tendencies == {}
#     assert "tendency_of_air_potential_temperature" in diagnostics
#     compare_dataarrays(
#         out,
#         diagnostics["tendency_of_air_potential_temperature"],
#         compare_coordinate_values=False,
#     )
#
#     #
#     # tendency_of_air_potential_temperature_on_interface_levels=True
#     # air_pressure_on_interface_levels=False
#     #
#     # tendency_of_air_potential_temperature_in_diagnostics=False
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=False,
#         tendency_of_air_potential_temperature_on_interface_levels=True,
#         air_pressure_on_interface_levels=False,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "air_potential_temperature" in psf.tendency_properties
#     tendencies, diagnostics = psf(state)
#     assert "air_potential_temperature" in tendencies
#     compare_dataarrays(
#         out, tendencies["air_potential_temperature"], compare_coordinate_values=False
#     )
#     assert diagnostics == {}
#
#     # tendency_of_air_potential_temperature_in_diagnostics=True
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=True,
#         tendency_of_air_potential_temperature_on_interface_levels=True,
#         air_pressure_on_interface_levels=False,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
#     tendencies, diagnostics = psf(state)
#     assert tendencies == {}
#     assert "tendency_of_air_potential_temperature" in diagnostics
#     compare_dataarrays(
#         out,
#         diagnostics["tendency_of_air_potential_temperature"],
#         compare_coordinate_values=False,
#     )
#
#     #
#     # tendency_of_air_potential_temperature_on_interface_levels=False
#     # air_pressure_on_interface_levels=True
#     #
#     if state["time"] > t0:
#         theta = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
#         p = 0.5 * (
#             state["air_pressure_on_interface_levels"].values[:, :, :-1]
#             + state["air_pressure_on_interface_levels"].values[:, :, 1:]
#         )
#         z = 0.5 * (
#             state["height_on_interface_levels"].values[:, :, :-1]
#             + state["height_on_interface_levels"].values[:, :, 1:]
#         )
#         h = state["height_on_interface_levels"].values[:, :, -1:]
#
#         out = (
#             theta
#             * rd
#             * a
#             / (p * cp)
#             * np.exp(-a * (z - h))
#             * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
#             * (x ** 2 + y ** 2 < cl ** 2)
#         )
#     else:
#         out = np.zeros((nx, ny, nz), dtype=dtype)
#
#     # tendency_of_air_potential_temperature_in_diagnostics=False
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=False,
#         tendency_of_air_potential_temperature_on_interface_levels=False,
#         air_pressure_on_interface_levels=True,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure_on_interface_levels" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "air_potential_temperature" in psf.tendency_properties
#     tendencies, diagnostics = psf(state)
#     assert "air_potential_temperature" in tendencies
#     compare_dataarrays(
#         out, tendencies["air_potential_temperature"], compare_coordinate_values=False
#     )
#     assert diagnostics == {}
#
#     # tendency_of_air_potential_temperature_in_diagnostics=True
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=True,
#         tendency_of_air_potential_temperature_on_interface_levels=False,
#         air_pressure_on_interface_levels=True,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure_on_interface_levels" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
#     tendencies, diagnostics = psf(state)
#     assert tendencies == {}
#     assert "tendency_of_air_potential_temperature" in diagnostics
#     compare_dataarrays(
#         out,
#         diagnostics["tendency_of_air_potential_temperature"],
#         compare_coordinate_values=False,
#     )
#
#     #
#     # tendency_of_air_potential_temperature_on_interface_levels=True
#     # air_pressure_on_interface_levels=True
#     #
#     if state["time"] > t0:
#         theta = grid.z_on_interface_levels.to_units("K").values[
#             np.newaxis, np.newaxis, :
#         ]
#         p = state["air_pressure_on_interface_levels"].values
#         z = state["height_on_interface_levels"].values
#         h = state["height_on_interface_levels"].values[:, :, -1:]
#
#         out = (
#             theta
#             * rd
#             * a
#             / (p * cp)
#             * np.exp(-a * (z - h))
#             * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
#             * (x ** 2 + y ** 2 < cl ** 2)
#         )
#     else:
#         out = np.zeros((nx, ny, nz + 1), dtype=dtype)
#
#     # tendency_of_air_potential_temperature_in_diagnostics=False
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=False,
#         tendency_of_air_potential_temperature_on_interface_levels=True,
#         air_pressure_on_interface_levels=True,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure_on_interface_levels" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert "air_potential_temperature_on_interface_levels" in psf.tendency_properties
#     tendencies, diagnostics = psf(state)
#     assert "air_potential_temperature_on_interface_levels" in tendencies
#     compare_dataarrays(
#         out,
#         tendencies["air_potential_temperature_on_interface_levels"],
#         compare_coordinate_values=False,
#     )
#     assert diagnostics == {}
#
#     # tendency_of_air_potential_temperature_in_diagnostics=True
#     psf = PrescribedSurfaceHeating(
#         domain,
#         tendency_of_air_potential_temperature_in_diagnostics=True,
#         tendency_of_air_potential_temperature_on_interface_levels=True,
#         air_pressure_on_interface_levels=True,
#         amplitude_at_day_sw=f0d_sw_da,
#         amplitude_at_day_fw=f0d_fw_da,
#         amplitude_at_night_sw=f0n_sw_da,
#         amplitude_at_night_fw=f0n_fw_da,
#         frequency_sw=w_sw_da,
#         frequency_fw=w_fw_da,
#         attenuation_coefficient_at_day=ad_da,
#         attenuation_coefficient_at_night=an_da,
#         characteristic_length=cl_da,
#         starting_time=t0,
#         backend=backend,
#     )
#     assert "air_pressure_on_interface_levels" in psf.input_properties
#     assert "height_on_interface_levels" in psf.input_properties
#     assert (
#         "tendency_of_air_potential_temperature_on_interface_levels"
#         in psf.diagnostic_properties
#     )
#     tendencies, diagnostics = psf(state)
#     assert tendencies == {}
#     assert "tendency_of_air_potential_temperature_on_interface_levels" in diagnostics
#     compare_dataarrays(
#         out,
#         diagnostics["tendency_of_air_potential_temperature_on_interface_levels"],
#         compare_coordinate_values=False,
#     )


if __name__ == "__main__":
    pytest.main([__file__])
    # test("upwind", "numpy", float, None)
