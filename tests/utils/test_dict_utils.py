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
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import pytest

from gt4py import storage as gt_storage

from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict, get_dataarray_3d

from tests.conf import (
    backend as conf_backend,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utilities import (
    compare_dataarrays,
    st_domain,
    st_floats,
    st_isentropic_state_f,
    st_one_of,
)


__field_properties = {
    "air_isentropic_density": {"units": "kg m^-2 K^-1"},
    "air_pressure_on_interface_levels": {"units": "hPa"},
    "exner_function_on_interface_levels": {"units": "J kg^-1 K^-1"},
    "height_on_interface_levels": {"units": "km"},
    "montgomery_potential": {"units": "m^2 s^-2"},
    "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    "x_velocity_at_u_locations": {"units": "km hr^-1"},
    "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    "y_velocity_at_v_locations": {"units": "km s^-1"},
}


def get_grid_shape(name, nx, ny, nz):
    return (
        nx + int("_at_u_locations" in name),
        ny + int("_at_v_locations" in name),
        nz + int("_on_interface_levels" in name),
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_copy(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="src",
    )
    dst = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dst",
    )

    keys = tuple(key for key in src.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            src.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dst.pop(key, None)

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in src if key in dst and key != "time")
    unshared_keys = tuple(key for key in src if key not in dst and key != "time")

    #
    # unshared_variables_in_output=False
    #
    dst_dc = deepcopy_dataarray_dict(dst)
    op.copy(dst, src, unshared_variables_in_output=False)
    for key in shared_keys:
        with subtests.test(key=key):
            compare_dataarrays(dst[key], src[key].to_units(dst[key].attrs["units"]))
    assert len(dst) == len(dst_dc)

    #
    # unshared_variables_in_output=True
    #
    dst_dcdc = deepcopy_dataarray_dict(dst_dc)
    op.copy(dst_dc, src, unshared_variables_in_output=True)
    for key in shared_keys:
        with subtests.test(key=key):
            compare_dataarrays(dst_dc[key], src[key].to_units(dst[key].attrs["units"]))
    for key in unshared_keys:
        with subtests.test(key=key):
            compare_dataarrays(dst_dc[key], src[key])
    assert len(dst_dc) == len(dst_dcdc) + len(unshared_keys)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_add(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="src",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dst",
    )

    out_a = deepcopy_dataarray_dict(dict1)
    out_b = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in out_a.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    unshared_keys = tuple(key for key in dict1 if key not in dict2 and key != "time")
    unshared_keys += tuple(key for key in dict2 if key not in dict1 and key != "time")

    #
    # out=None, unshared_variables_in_output=False
    #
    out = op.add(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)

    #
    # out=None, unshared_variables_in_output=True
    #
    out = op.add(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            _dict = dict1 if key in dict1 else dict2
            out_val = _dict[key].to_units(units)
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)

    #
    # out=out_a, unshared_variables_in_output=False
    #
    op.add(
        dict1,
        dict2,
        out=out_a,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)

    #
    # out=out_b, unshared_variables_in_output=True
    #
    op.add(
        dict1,
        dict2,
        out=out_b,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_b[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            _dict = dict1 if key in dict1 else dict2
            out_val = _dict[key].to_units(units)
            compare_dataarrays(out_b[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_iadd(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="src",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dst",
    )

    keys = tuple(key for key in dict1.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    dict1_a = deepcopy_dataarray_dict(dict1)
    dict1_b = deepcopy_dataarray_dict(dict1)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    unshared_keys = tuple(key for key in dict1 if key not in dict2 and key != "time")
    unshared_keys += tuple(key for key in dict2 if key not in dict1 and key != "time")

    #
    # unshared_variables_in_output=False
    #
    op.iadd(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1_a[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(dict1[key], out_val, compare_coordinate_values=False)

    #
    # unshared_variables_in_output=True
    #
    op.iadd(
        dict1_a,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1_b[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(dict1_a[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            _dict = dict1_b if key in dict1_b else dict2
            out_val = _dict[key].to_units(units)
            compare_dataarrays(dict1_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_sub(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="src",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dst",
    )

    out_a = deepcopy_dataarray_dict(dict1)
    out_b = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in out_a.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    unshared_keys = tuple(key for key in dict1 if key not in dict2 and key != "time")
    unshared_keys += tuple(key for key in dict2 if key not in dict1 and key != "time")

    #
    # out=None, unshared_variables_in_output=False
    #
    out = op.sub(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)

    #
    # out=None, unshared_variables_in_output=True
    #
    out = op.sub(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            if key in dict1:
                out_val = dict1[key].to_units(units)
                compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
            else:
                out_val = dict2[key].to_units(units)
                out_val.values *= -1
                compare_dataarrays(out[key], out_val, compare_coordinate_values=False)

    #
    # out=out_a, unshared_variables_in_output=False
    #
    op.sub(
        dict1,
        dict2,
        out=out_a,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)

    #
    # out=out_b, unshared_variables_in_output=True
    #
    op.sub(
        dict1,
        dict2,
        out=out_b,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_b[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            if key in dict1:
                out_val = dict1[key].to_units(units)
                compare_dataarrays(out_b[key], out_val, compare_coordinate_values=False)
            else:
                out_val = dict2[key].to_units(units)
                out_val.values *= -1
                compare_dataarrays(out_b[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_isub(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="src",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dst",
    )

    keys = tuple(key for key in dict1.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    dict1_a = deepcopy_dataarray_dict(dict1)
    dict1_b = deepcopy_dataarray_dict(dict1)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    unshared_keys = tuple(key for key in dict1 if key not in dict2 and key != "time")
    unshared_keys += tuple(key for key in dict2 if key not in dict1 and key != "time")

    #
    # unshared_variables_in_output=False
    #
    op.isub(
        dict1,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=False,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1_a[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(dict1[key], out_val, compare_coordinate_values=False)

    #
    # unshared_variables_in_output=True
    #
    op.isub(
        dict1_a,
        dict2,
        field_properties=field_properties_passed,
        unshared_variables_in_output=True,
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1_b[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 - field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(dict1_a[key], out_val, compare_coordinate_values=False)
    for key in unshared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            if key in dict1_b:
                out_val = dict1_b[key].to_units(units)
                compare_dataarrays(dict1_a[key], out_val, compare_coordinate_values=False)
            else:
                out_val = dict2[key].to_units(units)
                out_val.values *= -1
                compare_dataarrays(dict1_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_scale(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    out_a = deepcopy_dataarray_dict(dict1)

    field_properties = {}
    field_properties_passed = {}
    for key in dict1.keys():
        if key != "time":
            if data.draw(hyp_st.booleans()):
                field_properties[key] = __field_properties[key]
                field_properties_passed[key] = field_properties[key]
            else:
                field_properties[key] = {"units": dict1[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    #
    # out=None
    #
    out = op.scale(dict1, f, field_properties=field_properties_passed)
    for key in dict1.keys():
        with subtests.test(key=key):
            if key != "time":
                units = field_properties[key]["units"]
                field = dict1[key].to_units(units).values
                out_val = get_dataarray_3d(f * field, grid, units, set_coordinates=False)
                compare_dataarrays(out[key], out_val, compare_coordinate_values=False)

    #
    # out=out_a
    #
    op.scale(dict1, f, out=out_a, field_properties=field_properties_passed)
    for key in dict1.keys():
        with subtests.test(key=key):
            if key != "time":
                units = field_properties[key]["units"]
                field = dict1[key].to_units(units).values
                out_val = get_dataarray_3d(f * field, grid, units, set_coordinates=False)
                compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_iscale(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    dict1_dc = deepcopy_dataarray_dict(dict1)

    field_properties = {}
    field_properties_passed = {}
    for key in dict1.keys():
        if key != "time":
            if data.draw(hyp_st.booleans()):
                field_properties[key] = __field_properties[key]
                field_properties_passed[key] = field_properties[key]
            else:
                field_properties[key] = {"units": dict1[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    op.iscale(dict1, f, field_properties=field_properties_passed)

    for key in dict1.keys():
        with subtests.test(key=key):
            if key != "time":
                units = field_properties[key]["units"]
                field = dict1_dc[key].to_units(units).values
                out_val = get_dataarray_3d(f * field, grid, units, set_coordinates=False)
                compare_dataarrays(dict1[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_addsub(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict2",
    )
    dict3 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict3",
    )

    out_a = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in dict1.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict3.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        else:
            if key in dict1:
                field_properties[key] = {"units": dict1[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    shared_keys = tuple(key for key in shared_keys if key in dict3)

    #
    # out=None
    #
    out = op.addsub(dict1, dict2, dict3, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2 - field3, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    assert len(out) == len(shared_keys) + 1

    #
    # out=out_a
    #
    op.addsub(dict1, dict2, dict3, out=out_a, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2 - field3, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_iaddsub(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict2",
    )
    dict3 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict3",
    )

    dict1_a = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in dict1.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict3.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        else:
            if key in dict1:
                field_properties[key] = {"units": dict1[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
    shared_keys = tuple(key for key in shared_keys if key in dict3)

    op.iaddsub(dict1, dict2, dict3, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1_a[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + field2 - field3, grid, units, set_coordinates=False
            )
            compare_dataarrays(dict1[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_fma(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict2",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    out_a = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in out_a.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")

    #
    # out=None
    #
    out = op.fma(dict1, dict2, f, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + f * field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    assert len(out) == len(shared_keys) + 1

    #
    # out=out_a
    #
    op.fma(dict1, dict2, f, out=out_a, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_val = get_dataarray_3d(
                field1 + f * field2, grid, units, set_coordinates=False
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_sts_rk2_0(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict2",
    )
    dict3 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict3",
    )
    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    out_a = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in out_a.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")

    dt = timestep.total_seconds()

    #
    # out=None
    #
    out = op.sts_rk2_0(dt, dict1, dict2, dict3, field_properties=field_properties_passed)
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                0.5 * (field1 + field2 + dt * field3), grid, units, set_coordinates=False
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    assert len(out) == len(shared_keys) + 1

    #
    # out=out_a
    #
    op.sts_rk2_0(
        dt, dict1, dict2, dict3, out=out_a, field_properties=field_properties_passed
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                0.5 * (field1 + field2 + dt * field3), grid, units, set_coordinates=False
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_sts_rk3ws_0(data, subtests):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="domain",
    )
    grid = domain.physical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    dict1 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict1",
    )
    dict2 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict2",
    )
    dict3 = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, default_origin=default_origin
        ),
        label="dict3",
    )
    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    out_a = deepcopy_dataarray_dict(dict1)

    keys = tuple(key for key in out_a.keys() if key != "time")
    for key in keys:
        if data.draw(hyp_st.booleans()):
            dict1.pop(key, None)
        if data.draw(hyp_st.booleans()):
            dict2.pop(key, None)

    field_properties = {}
    field_properties_passed = {}
    for key in keys:
        if data.draw(hyp_st.booleans()):
            field_properties[key] = __field_properties[key]
            field_properties_passed[key] = field_properties[key]
        elif key in dict1:
            field_properties[key] = {"units": dict1[key].attrs["units"]}
        elif key in dict2:
            field_properties[key] = {"units": dict2[key].attrs["units"]}

    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")

    # ========================================
    # test bed
    # ========================================
    op = DataArrayDictOperator(
        gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=False
    )

    shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")

    dt = timestep.total_seconds()

    #
    # out=None
    #
    out = op.sts_rk3ws_0(
        dt, dict1, dict2, dict3, field_properties=field_properties_passed
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                (2.0 * field1 + field2 + dt * field3) / 3.0,
                grid,
                units,
                set_coordinates=False,
            )
            compare_dataarrays(out[key], out_val, compare_coordinate_values=False)
    assert len(out) == len(shared_keys) + 1

    #
    # out=out_a
    #
    op.sts_rk3ws_0(
        dt, dict1, dict2, dict3, out=out_a, field_properties=field_properties_passed
    )
    for key in shared_keys:
        with subtests.test(key=key):
            units = field_properties[key]["units"]
            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            field3 = dict3[key].to_units(units).values
            out_val = get_dataarray_3d(
                (2.0 * field1 + field2 + dt * field3) / 3.0,
                grid,
                units,
                set_coordinates=False,
            )
            compare_dataarrays(out_a[key], out_val, compare_coordinate_values=False)


if __name__ == "__main__":
    pytest.main([__file__])
