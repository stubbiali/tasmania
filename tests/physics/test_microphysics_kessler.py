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
from datetime import timedelta
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
from property_cached import cached_property
import pytest

from sympl import DataArray

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustmentDiagnostic,
    KesslerSaturationAdjustmentPrognostic,
    KesslerSedimentation,
)
from tasmania import get_dataarray_3d
from tasmania.python.utils.meteo import tetens_formula

from tests import conf
from tests.physics.test_microphysics_utils import flux_properties
from tests.strategies import (
    st_one_of,
    st_domain,
    st_isentropic_state_f,
)
from tests.suites import (
    DiagnosticComponentTestSuite,
    DomainSuite,
    TendencyComponentTestSuite,
)
from tests.utilities import compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def get_state(test_suite):
    state = test_suite.hyp_data.draw(
        st_isentropic_state_f(
            test_suite.ds.grid,
            moist=True,
            backend=test_suite.ds.backend,
            storage_shape=test_suite.storage_shape,
            storage_options=test_suite.ds.so,
        ),
        label="state",
    )

    if not test_suite.apoif:
        storage_shape = (
            test_suite.storage_shape
            or test_suite.component.get_field_storage_shape("air_pressure")
        )
        nx, ny, nz = (
            test_suite.ds.grid.nx,
            test_suite.ds.grid.ny,
            test_suite.ds.grid.nz,
        )

        p_np = to_numpy(
            state["air_pressure_on_interface_levels"].to_units("Pa").data
        )[:nx, :ny, : nz + 1]
        p_unstg = test_suite.component.zeros(shape=storage_shape)
        p_unstg[:nx, :ny, :nz] = test_suite.component.as_storage(
            data=0.5 * (p_np[:, :, :-1] + p_np[:, :, 1:]),
        )
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            test_suite.ds.grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn_np = to_numpy(
            state["exner_function_on_interface_levels"]
            .to_units("J kg^-1 K^-1")
            .data
        )[:nx, :ny, : nz + 1]
        exn_unstg = test_suite.component.zeros(shape=storage_shape)
        exn_unstg[:nx, :ny, :nz] = test_suite.component.as_storage(
            data=0.5 * (exn_np[:, :, :-1] + exn_np[:, :, 1:]),
        )
        state["exner_function"] = get_dataarray_3d(
            exn_unstg,
            test_suite.ds.grid,
            "J kg^-1 K^-1",
            name="exner_function",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

    return state


class KesslerMicrophysicsTestSuite(TendencyComponentTestSuite):
    def __init__(self, domain_suite):
        self.apoif = domain_suite.hyp_data.draw(
            hyp_st.booleans(), label="apoif"
        )
        # self.apiif = True
        self.toaptid = domain_suite.hyp_data.draw(
            hyp_st.booleans(), label="toaptid"
        )
        # self.toaptid = False
        self.re = domain_suite.hyp_data.draw(hyp_st.booleans(), label="re")
        # self.re = False
        self.a = domain_suite.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=10), label="a"
        )
        self.k1 = domain_suite.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=10), label="k1"
        )
        self.k2 = domain_suite.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=10), label="k2"
        )

        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return KesslerMicrophysics(
            self.ds.domain,
            self.ds.grid_type,
            air_pressure_on_interface_levels=self.apoif,
            tendency_of_air_potential_temperature_in_diagnostics=self.toaptid,
            rain_evaporation=self.re,
            autoconversion_threshold=DataArray(
                self.a, attrs={"units": "g g^-1"}
            ),
            autoconversion_rate=DataArray(self.k1, attrs={"units": "s^-1"}),
            collection_rate=DataArray(self.k2, attrs={"units": "s^-1"}),
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return get_state(self)

    def get_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        i, j, k = slice(0, nx), slice(0, ny), slice(0, nz)

        qc = raw_state_np[mfcw]
        qr = raw_state_np[mfpw]

        ar = self.k1 * (qc - self.a) * (qc > self.a)
        cr = self.k2 * qc * np.where(qr > 0, qr ** 0.875, 0)

        tendencies, diagnostics = {}, {}
        tendencies[mfcw] = -ar - cr
        tendencies[mfpw] = ar + cr

        if self.re:
            rho = raw_state_np["air_density"]
            t = raw_state_np["air_temperature"]
            qv = raw_state_np[mfwv]
            if not self.apoif:
                p = raw_state_np["air_pressure"]
                exn = raw_state_np["exner_function"]
            else:
                tmp = raw_state_np["air_pressure_on_interface_levels"]
                p = np.zeros_like(rho)
                p[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])
                tmp = raw_state_np["exner_function_on_interface_levels"]
                exn = np.zeros_like(rho)
                exn[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])

            rd = 287.0
            rv = 461.5
            lhvw = 2.5e6
            beta = rd / rv

            ps = tetens_formula(t)
            qvs = np.zeros_like(ps)
            qvs[i, j, k] = beta * ps[i, j, k] / p[i, j, k]
            er = np.where(
                qr > 0.0,
                0.0484794 * (qvs - qv) * (rho * qr) ** (13.0 / 20.0),
                0.0,
            )
            # er[qr < 0] = 0.0
            tendencies[mfwv] = er
            tendencies[mfpw] -= er

            if self.toaptid:
                diagnostics["tendency_of_air_potential_temperature"] = (
                    -lhvw / exn * er
                )
            else:
                tendencies["air_potential_temperature"] = -lhvw / exn * er

        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_microphysics(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = KesslerMicrophysicsTestSuite(ds)
    ts.run()


class KesslerSaturationAdjustmentDiagnosticTestSuite(
    TendencyComponentTestSuite
):
    def __init__(self, domain_suite):
        self.apoif = domain_suite.hyp_data.draw(
            hyp_st.booleans(), label="apoif"
        )
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return KesslerSaturationAdjustmentDiagnostic(
            self.ds.domain,
            self.ds.grid_type,
            air_pressure_on_interface_levels=self.apoif,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return get_state(self)

    def get_tendencies_and_diagnostics(self, raw_state_np, dt):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        i, j, k = slice(0, nx), slice(0, ny), slice(0, nz)

        t = raw_state_np["air_temperature"]
        qv = raw_state_np[mfwv]
        qc = raw_state_np[mfcw]
        if not self.apoif:
            p = raw_state_np["air_pressure"]
            exn = raw_state_np["exner_function"]
        else:
            tmp = raw_state_np["air_pressure_on_interface_levels"]
            p = np.zeros_like(t)
            p[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])
            tmp = raw_state_np["exner_function_on_interface_levels"]
            exn = np.zeros_like(t)
            exn[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])

        rd = 287.0
        rv = 461.5
        cp = 1004.0
        lhvw = 2.5e6
        beta = rd / rv

        ps = tetens_formula(t)
        qvs = np.zeros_like(ps)
        qvs[i, j, k] = beta * ps[i, j, k] / p[i, j, k]
        sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
        dq = qc.copy()
        dq[sat <= qc] = sat[sat <= qc]

        tendencies = {
            "air_potential_temperature": (lhvw / exn) * (-dq / dt),
        }
        diagnostics = {
            mfwv: qv + dq,
            mfcw: qc - dq,
            "air_temperature": t - lhvw * dq / cp,
        }

        return tendencies, diagnostics


# def kessler_saturation_adjustment_diagnostic_validation(
#     timestep, p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv
# ):
#     p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
#     exn = (
#         exn
#         if exn.shape[2] == t.shape[2]
#         else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
#     )
#
#     ps = svpf(t)
#     qvs = beta * ps / p
#     sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
#     dq = deepcopy(qc)
#     dq[sat <= qc] = sat[sat <= qc]
#     dt = -lhvw * dq / cp
#     cv = -dq / timestep
#
#     return qv + dq, qc - dq, t + dt, lhvw / exn * cv


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_saturation_adjustment_diagnostic(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = KesslerSaturationAdjustmentDiagnosticTestSuite(ds)
    ts.run()


class KesslerSaturationAdjustmentPrognosticTestSuite(
    TendencyComponentTestSuite
):
    def __init__(self, domain_suite):
        self.apoif = domain_suite.hyp_data.draw(
            hyp_st.booleans(), label="apoif"
        )
        self.sr = domain_suite.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=1), label="sr"
        )

        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return KesslerSaturationAdjustmentPrognostic(
            self.ds.domain,
            self.ds.grid_type,
            air_pressure_on_interface_levels=self.apoif,
            saturation_rate=DataArray(self.sr, attrs={"units": "s^-1"}),
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return get_state(self)

    def get_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        i, j, k = slice(0, nx), slice(0, ny), slice(0, nz)

        t = raw_state_np["air_temperature"]
        qv = raw_state_np[mfwv]
        qc = raw_state_np[mfcw]
        if not self.apoif:
            p = raw_state_np["air_pressure"]
            exn = raw_state_np["exner_function"]
        else:
            tmp = raw_state_np["air_pressure_on_interface_levels"]
            p = np.zeros_like(t)
            p[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])
            tmp = raw_state_np["exner_function_on_interface_levels"]
            exn = np.zeros_like(t)
            exn[i, j, k] = 0.5 * (tmp[i, j, :nz] + tmp[i, j, 1 : nz + 1])

        rd = 287.0
        rv = 461.5
        cp = 1004.0
        lhvw = 2.5e6
        beta = rd / rv

        ps = tetens_formula(t)
        qvs = np.zeros_like(ps)
        qvs[i, j, k] = beta * ps[i, j, k] / p[i, j, k]
        sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
        dq = qc.copy()
        dq[sat <= qc] = sat[sat <= qc]

        tendencies = {
            mfwv: self.sr * dq,
            mfcw: -self.sr * dq,
            "air_potential_temperature": -self.sr * (lhvw / exn) * dq,
        }
        diagnostics = {}

        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_saturation_adjustment_prognostic(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = KesslerSaturationAdjustmentPrognosticTestSuite(ds)
    ts.run()


class KesslerFallVelocityTestSuite(DiagnosticComponentTestSuite):
    @cached_property
    def component(self):
        return KesslerFallVelocity(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=True,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_diagnostics(self, raw_state_np):
        nz = self.ds.grid.nz
        rho = raw_state_np["air_density"]
        qr = raw_state_np[mfpw]
        return {
            "raindrop_fall_velocity": 36.34
            * (0.001 * rho * qr) ** 0.1346
            * np.sqrt(rho[:, :, nz - 1 : nz] / rho)
        }


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_fall_velocity(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = KesslerSaturationAdjustmentPrognosticTestSuite(ds)
    ts.run()


class KesslerSedimentationTestSuite(TendencyComponentTestSuite):
    def __init__(self, domain_suite, flux_scheme):
        self.flux_scheme = flux_scheme
        self.maxcfl = domain_suite.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=1), label="maxcfl"
        )
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return KesslerSedimentation(
            self.ds.domain,
            self.ds.grid_type,
            self.flux_scheme,
            self.maxcfl,
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
                precipitation=True,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )
        rfv = KesslerFallVelocity(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )
        state.update(rfv(state))
        return state

    def get_tendencies_and_diagnostics(self, raw_state_np, dt):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        rho = raw_state_np["air_density"]
        h = raw_state_np["height_on_interface_levels"]
        qr = raw_state_np[mfpw]
        vt = raw_state_np["raindrop_fall_velocity"]

        # dh = h[:, :, :-1] - h[:, :, 1:]
        # ids = np.where(vt > self.maxcfl * dh / dt)
        # vt[ids] = self.maxcfl * dh[ids] / dt

        out = np.zeros_like(raw_state_np[mfpw])
        out[:nx, :ny, :nz] = (
            flux_properties[self.flux_scheme]["validation"](
                rho[:nx, :ny, :nz],
                h[:nx, :ny, : nz + 1],
                qr[:nx, :ny, :nz],
                vt[:nx, :ny, :nz],
                staggering=True,
            )
            / rho[:nx, :ny, :nz]
        )

        tendencies = {mfpw: out}
        diagnostics = {}

        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("flux_scheme", flux_properties.keys())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_sedimentation(data, flux_scheme, backend, dtype):
    ds = DomainSuite(data, backend, dtype, zaxis_length=(3, 50))
    ts = KesslerSedimentationTestSuite(ds, flux_scheme)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
