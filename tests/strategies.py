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
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
from hypothesis import assume, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
from pandas import Timedelta
from pint import UnitRegistry
from sympl import DataArray
from sympl._core.units import clean_units

import tasmania as taz
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.data import get_physical_constants
from tasmania.python.utils.storage import (
    get_dataarray_2d,
    get_dataarray_3d,
    get_default_origin,
    zeros,
)
from tasmania.python.utils.utils import equal_to
from tasmania.python.utils.backend import is_gt

from tests import conf
from tests.utilities import get_interval, get_nanoseconds, pi_function


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


default_physical_constants = {
    "gas_constant_of_dry_air": DataArray(
        287.05, attrs={"units": "J K^-1 kg^-1"}
    ),
    "gravitational_acceleration": DataArray(9.81, attrs={"units": "m s^-2"}),
    "reference_air_pressure": DataArray(1.0e5, attrs={"units": "Pa"}),
    "specific_heat_of_dry_air_at_constant_pressure": DataArray(
        1004.0, attrs={"units": "J K^-1 kg^-1"}
    ),
}


unit_registry = UnitRegistry()


def st_floats(**kwargs):
    """ Strategy drawing a non-nan and non-infinite floats. """
    kwargs.pop("allow_nan", None)
    kwargs.pop("allow_infinite", None)
    return hyp_st.floats(allow_nan=False, allow_infinity=False, **kwargs)


def st_one_of(seq):
    """ Strategy drawing one of the elements of the input sequence. """
    return hyp_st.one_of(hyp_st.just(el) for el in seq)


@hyp_st.composite
def st_default_origin(
    draw, storage_shape, min_default_origin=None, max_default_origin=None
):
    default_origin = draw(st_one_of(conf.default_origin))
    return get_default_origin(
        default_origin, storage_shape, min_default_origin, max_default_origin
    )


@hyp_st.composite
def st_timedeltas(draw, min_value, max_value):
    """ Strategy drawing a :class:`pandas.Timedelta`. """
    min_secs = min_value.total_seconds()
    max_secs = max_value.total_seconds()

    secs = draw(st_floats(min_value=min_secs, max_value=max_secs))
    nanosecs = get_nanoseconds(secs)

    return Timedelta(seconds=secs, nanoseconds=nanosecs)


@hyp_st.composite
def st_interface(draw, domain_z):
    """
    Strategy drawing a valid interface altitude where terrain-following
    grid surfaces flat back to horizontal surfaces.
    """
    min_value = np.min(domain_z.values)
    max_value = np.max(domain_z.values)
    zi_v = draw(st_floats(min_value=min_value, max_value=max_value))
    return DataArray(zi_v, attrs={"units": domain_z.attrs["units"]})


@hyp_st.composite
def st_interval(draw, *, axis_name="x"):
    """
    Strategy drawing an interval along the `axis_name`-axis according to the
    specifications defined in `conf.py`.
    """
    axis_properties = eval("conf.axis_{}".format(axis_name))
    units = draw(st_one_of(axis_properties["units_to_range"].keys()))

    el0 = draw(
        st_floats(
            min_value=axis_properties["units_to_range"][units][0],
            max_value=axis_properties["units_to_range"][units][1],
        )
    )
    el1 = draw(
        st_floats(
            min_value=axis_properties["units_to_range"][units][0],
            max_value=axis_properties["units_to_range"][units][1],
        )
    )
    assume(not equal_to(el0, el1))

    return get_interval(
        el0,
        el1,
        draw(st_one_of(axis_properties["dims"])),
        draw(hyp_st.just(units)),
        draw(hyp_st.just(axis_properties["increasing"])),
    )


def st_length(axis_name="x", min_value=None, max_value=None):
    """ Strategy drawing the number of grid points in the `axis_name`-direction. """
    axis_properties = eval("conf.axis_{}".format(axis_name))
    min_val = axis_properties["length"][0] if min_value is None else min_value
    max_val = axis_properties["length"][1] if max_value is None else max_value
    return hyp_st.integers(min_value=min_val, max_value=max_val)


@hyp_st.composite
def st_physical_horizontal_grid(
    draw,
    *,
    xaxis_name="x",
    xaxis_length=None,
    yaxis_name="y",
    yaxis_length=None,
    dtype=np.float64
):
    """ Strategy drawing a :class:`tasmania.PhysicalHorizontalGrid` object. """
    domain_x = draw(st_interval(axis_name=xaxis_name))
    nx = draw(
        st_length(axis_name=xaxis_name)
        if xaxis_length is None
        else st_length(
            axis_name=xaxis_name,
            min_value=xaxis_length[0],
            max_value=xaxis_length[1],
        )
    )

    domain_y = draw(st_interval(axis_name=yaxis_name))
    ny = draw(
        st_length(axis_name=yaxis_name)
        if yaxis_length is None
        else st_length(
            axis_name=yaxis_name,
            min_value=yaxis_length[0],
            max_value=yaxis_length[1],
        )
    )

    assume(not (nx == 1 and ny == 1))

    return taz.PhysicalHorizontalGrid(domain_x, nx, domain_y, ny, dtype=dtype)


@hyp_st.composite
def st_topography_kwargs(
    draw,
    x,
    y,
    *,
    topography_type=None,
    time=None,
    max_height=None,
    center_x=None,
    center_y=None,
    topo_half_width_x=None,
    topo_half_width_y=None,
    expression=None,
    smooth=None
):
    """
    Strategy drawing a set of keyword arguments accepted by the constructor
    of :class:`tasmania.Topography`.
    """
    if topography_type is not None and isinstance(topography_type, str):
        _type = topography_type
    else:
        _type = draw(st_one_of(conf.topography["type"]))

    if time is not None and isinstance(time, Timedelta):
        _time = time
    elif draw(hyp_st.booleans()):
        _time = draw(
            st_timedeltas(
                min_value=conf.topography["time"][0],
                max_value=conf.topography["time"][1],
            )
        )
    else:
        _time = None

    if (
        max_height is not None
        and isinstance(max_height, DataArray)
        and max_height.shape == ()
    ):
        _max_height = max_height
    elif draw(hyp_st.booleans()):
        units = draw(st_one_of(conf.topography["units_to_max_height"].keys()))
        val = draw(
            st_floats(
                min_value=conf.topography["units_to_max_height"][units][0],
                max_value=conf.topography["units_to_max_height"][units][1],
            )
        )
        _max_height = DataArray(val, attrs={"units": units})
    else:
        _max_height = None

    if (
        center_x is not None
        and isinstance(center_x, DataArray)
        and center_x.shape == ()
    ):
        _center_x = center_x
    elif draw(hyp_st.booleans()):
        val = draw(
            st_floats(min_value=np.min(x.values), max_value=np.max(x.values))
        )
        _center_x = DataArray(val, attrs={"units": x.attrs["units"]})
    else:
        _center_x = None

    if (
        center_y is not None
        and isinstance(center_y, DataArray)
        and center_y.shape == ()
    ):
        _center_y = center_y
    elif draw(hyp_st.booleans()):
        val = draw(
            st_floats(min_value=np.min(y.values), max_value=np.max(y.values))
        )
        _center_y = DataArray(val, attrs={"units": y.attrs["units"]})
    else:
        _center_y = None

    if (
        topo_half_width_x is not None
        and isinstance(topo_half_width_x, DataArray)
        and topo_half_width_x.shape == ()
    ):
        _topo_half_width_x = topo_half_width_x
    elif draw(hyp_st.booleans()):
        units = draw(
            st_one_of(conf.topography["units_to_half_width_x"].keys())
        )
        val = draw(
            st_floats(
                min_value=conf.topography["units_to_half_width_x"][units][0],
                max_value=conf.topography["units_to_half_width_x"][units][1],
            )
        )
        _topo_half_width_x = DataArray(val, attrs={"units": units})
    else:
        _topo_half_width_x = None

    if (
        topo_half_width_y is not None
        and isinstance(topo_half_width_y, DataArray)
        and topo_half_width_y.shape == ()
    ):
        _topo_half_width_y = topo_half_width_y
    elif draw(hyp_st.booleans()):
        units = draw(
            st_one_of(conf.topography["units_to_half_width_y"].keys())
        )
        val = draw(
            st_floats(
                min_value=conf.topography["units_to_half_width_y"][units][0],
                max_value=conf.topography["units_to_half_width_y"][units][1],
            )
        )
        _topo_half_width_y = DataArray(val, attrs={"units": units})
    else:
        _topo_half_width_y = None

    if expression is not None and isinstance(expression, str):
        _expression = expression
    elif draw(hyp_st.booleans()):
        _expression = draw(st_one_of(conf.topography["str"]))
    else:
        _expression = None

    if smooth is not None and isinstance(smooth, bool):
        _smooth = smooth
    else:
        _smooth = draw(hyp_st.booleans())

    kwargs = {
        "type": _type,
        "time": _time,
        "max_height": _max_height,
        "center_x": _center_x,
        "center_y": _center_y,
        "width_x": _topo_half_width_x,
        "width_y": _topo_half_width_y,
        "expression": _expression,
        "smooth": _smooth,
    }

    return kwargs


@hyp_st.composite
def st_physical_grid(
    draw,
    *,
    xaxis_name="x",
    xaxis_length=None,
    yaxis_name="y",
    yaxis_length=None,
    zaxis_name="z",
    zaxis_length=None,
    dtype=np.float64
):
    """ Strategy drawing a :class:`tasmania.PhysicalGrid` object. """
    nx = draw(
        st_length(axis_name=xaxis_name)
        if xaxis_length is None
        else st_length(
            axis_name=xaxis_name,
            min_value=xaxis_length[0],
            max_value=xaxis_length[1],
        )
    )
    ny = draw(
        st_length(axis_name=yaxis_name)
        if yaxis_length is None
        else st_length(
            axis_name=yaxis_name,
            min_value=yaxis_length[0],
            max_value=yaxis_length[1],
        )
    )

    assume(not (nx == 1 and ny == 1))

    domain_x = draw(st_interval(axis_name=xaxis_name))
    domain_y = draw(st_interval(axis_name=yaxis_name))

    domain_z = draw(st_interval(axis_name=zaxis_name))
    nz = draw(
        st_length(axis_name=zaxis_name)
        if zaxis_length is None
        else st_length(
            axis_name=zaxis_name,
            min_value=zaxis_length[0],
            max_value=zaxis_length[1],
        )
    )

    topo_kwargs = draw(st_topography_kwargs(domain_x, domain_y))
    topography_type = topo_kwargs.pop("type")

    return taz.PhysicalGrid(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        topography_type=topography_type,
        topography_kwargs=topo_kwargs,
        dtype=dtype,
    )


def st_horizontal_boundary_type():
    """ Strategy drawing a valid horizontal boundary type. """
    return st_one_of(conf.horizontal_boundary_types)


def st_horizontal_boundary_layers(nx, ny):
    """ Strategy drawing a valid number of boundary layers. """
    if ny == 1:
        return hyp_st.integers(min_value=1, max_value=min(3, int(nx / 2)))
    elif nx == 1:
        return hyp_st.integers(min_value=1, max_value=min(3, int(ny / 2)))
    else:
        return hyp_st.integers(
            min_value=1, max_value=min(3, min(int(nx / 2), int(ny / 2)))
        )


@hyp_st.composite
def st_horizontal_boundary_kwargs(draw, hb_type, nx, ny, nb, nz=None):
    """
    Strategy drawing a valid set of keyword arguments for the constructor
    of the specified class handling the lateral boundaries.
    """
    hb_kwargs = {}

    if hb_type == "relaxed":
        if ny == 1:
            nr = draw(
                hyp_st.integers(min_value=nb, max_value=min(8, int(nx / 2)))
            )
        elif nx == 1:
            nr = draw(
                hyp_st.integers(min_value=nb, max_value=min(8, int(ny / 2)))
            )
        else:
            nr = draw(
                hyp_st.integers(
                    min_value=nb,
                    max_value=min(8, min(int(nx / 2), int(ny / 2))),
                )
            )
        hb_kwargs["nr"] = nr
        hb_kwargs["nz"] = nz
    elif hb_type == "dirichlet":
        hb_kwargs["core"] = pi_function

    return hb_kwargs


@hyp_st.composite
def st_horizontal_boundary(
    draw,
    nx,
    ny,
    nb=None,
    nz=None,
    *,
    backend="numpy",
    backend_opts=None,
    dtype=np.float64,
    build_info=None,
    exec_info=None,
    default_origin=None,
    rebuild=False,
    storage_shape=None,
    managed_memory=False
):
    """ Strategy drawing an object handling the lateral boundary conditions. """
    hb_type = draw(st_horizontal_boundary_type())
    nb = nb if nb is not None else draw(st_horizontal_boundary_layers(nx, ny))
    hb_kwargs = draw(st_horizontal_boundary_kwargs(hb_type, nx, ny, nb, nz=nz))
    bo = BackendOptions(
        backend_opts=backend_opts,
        build_info=build_info,
        exec_info=exec_info,
        rebuild=rebuild,
    )
    so = StorageOptions(
        dtype=dtype,
        default_origin=default_origin,
        managed_memory=managed_memory,
    )
    return taz.HorizontalBoundary.factory(
        hb_type,
        nx,
        ny,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
        **hb_kwargs
    )


@hyp_st.composite
def st_domain(
    draw,
    xaxis_name="x",
    xaxis_length=None,
    yaxis_name="y",
    yaxis_length=None,
    zaxis_name="z",
    zaxis_length=None,
    nb=None,
    *,
    backend="numpy",
    backend_opts=None,
    dtype=np.float64,
    build_info=None,
    exec_info=None,
    default_origin=None,
    rebuild=False,
    storage_shape=None,
    managed_memory=False
):
    """ Strategy drawing a :class:`tasmania.Domain` object. """
    domain_x = draw(st_interval(axis_name=xaxis_name))
    nx = draw(
        st_length(axis_name=xaxis_name)
        if xaxis_length is None
        else st_length(
            axis_name=xaxis_name,
            min_value=xaxis_length[0],
            max_value=xaxis_length[1],
        )
    )

    domain_y = draw(st_interval(axis_name=yaxis_name))
    ny = draw(
        st_length(axis_name=yaxis_name)
        if yaxis_length is None
        else st_length(
            axis_name=yaxis_name,
            min_value=yaxis_length[0],
            max_value=yaxis_length[1],
        )
    )

    assume(not (nx == 1 and ny == 1))

    domain_z = draw(st_interval(axis_name=zaxis_name))
    nz = draw(
        st_length(axis_name=zaxis_name)
        if zaxis_length is None
        else st_length(
            axis_name=zaxis_name,
            min_value=zaxis_length[0],
            max_value=zaxis_length[1],
        )
    )

    hb_type = draw(st_horizontal_boundary_type())
    nb = nb if nb is not None else draw(st_horizontal_boundary_layers(nx, ny))
    if nx > 1:
        assume(nb < nx / 2)
    if ny > 1:
        assume(nb < ny / 2)
    hb_kwargs = draw(st_horizontal_boundary_kwargs(hb_type, nx, ny, nb, nz=nz))

    topo_kwargs = draw(st_topography_kwargs(domain_x, domain_y))
    topography_type = topo_kwargs.pop("type")

    bo = BackendOptions(
        backend_opts=backend_opts,
        build_info=build_info,
        exec_info=exec_info,
        rebuild=rebuild,
    )
    so = StorageOptions(
        dtype=dtype,
        default_origin=default_origin,
        managed_memory=managed_memory,
    )

    return taz.Domain(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        horizontal_boundary_type=hb_type,
        nb=nb,
        horizontal_boundary_kwargs=hb_kwargs,
        topography_type=topography_type,
        topography_kwargs=topo_kwargs,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )


@hyp_st.composite
def st_raw_field(
    draw,
    shape,
    min_value,
    max_value,
    *,
    backend="numpy",
    dtype=np.float64,
    default_origin=None
):
    """ Strategy drawing a random :class:`numpy.ndarray`. """
    storage = zeros(
        shape, backend=backend, dtype=dtype, default_origin=default_origin
    )
    storage[...] = draw(
        st_arrays(
            dtype,
            shape,
            elements=st_floats(min_value=min_value, max_value=max_value),
            fill=None,  # fill=hyp_st.nothing(),
        )
    )
    return storage


@hyp_st.composite
def st_horizontal_field(
    draw,
    grid,
    min_value,
    max_value,
    units,
    name,
    storage_shape=None,
    grid_origin=None,
    grid_shape=None,
    *,
    backend="numpy",
    default_origin=None,
    set_coordinates=True
):
    """ Strategy drawing a random field for the 2-D variable `field_name`. """
    storage_shape = storage_shape or (grid.nx, grid.ny)
    grid_origin = grid_origin or (0, 0)
    grid_shape = grid_shape or storage_shape
    dtype = grid.x.dtype

    storage = draw(
        st_raw_field(
            storage_shape,
            min_value,
            max_value,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    return get_dataarray_2d(
        storage,
        grid,
        units,
        name=name,
        grid_origin=grid_origin,
        grid_shape=grid_shape,
        set_coordinates=set_coordinates,
    )


@hyp_st.composite
def st_field(
    draw,
    grid,
    properties_name,
    name,
    storage_shape=None,
    grid_origin=None,
    grid_shape=None,
    *,
    backend="numpy",
    default_origin=None,
    set_coordinates=True
):
    """ Strategy drawing a random field for the variable `field_name`. """
    properties_dict = eval("conf.{}".format(properties_name))
    units = draw(st_one_of(properties_dict[name].keys()))

    storage_shape = storage_shape or (grid.nx, grid.ny, grid.nz)
    grid_origin = grid_origin or (0, 0, 0)
    grid_shape = grid_shape or storage_shape
    dtype = grid.x.dtype

    storage = draw(
        st_raw_field(
            storage_shape,
            properties_dict[name][units][0],
            properties_dict[name][units][1],
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    da = get_dataarray_3d(
        storage,
        grid,
        units,
        name=name,
        grid_origin=grid_origin,
        grid_shape=grid_shape,
        set_coordinates=set_coordinates,
    )

    return da


@hyp_st.composite
def st_state(
    draw,
    grid,
    time=None,
    *,
    backend="numpy",
    default_origin=None,
    storage_shape=None
):
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    dtype = grid.x.dtype

    if is_gt(backend) and storage_shape is None:
        storage_shape = (nx + 1, ny + 1, nz + 1)
    elif storage_shape is not None:
        storage_shape = (
            max(nx + 1, storage_shape[0]),
            max(ny + 1, storage_shape[1]),
            max(nz + 1, storage_shape[2]),
        )

    return_dict = {}

    # time
    return_dict["time"] = time or draw(hyp_st.datetimes())

    # (nx, ny, nz)
    field = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["afield"] = get_dataarray_3d(
        field, grid, "m", grid_shape=(nx, ny, nz), set_coordinates=False
    )

    # (nx+1, ny, nz)
    field = draw(
        st_raw_field(
            storage_shape or (nx + 1, ny, nz),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["afield_at_u_locations"] = get_dataarray_3d(
        field, grid, "kg", grid_shape=(nx + 1, ny, nz), set_coordinates=False
    )

    # (nx, ny+1, nz)
    field = draw(
        st_raw_field(
            storage_shape or (nx, ny + 1, nz),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["afield_at_v_locations"] = get_dataarray_3d(
        field, grid, "hr", grid_shape=(nx, ny + 1, nz), set_coordinates=False
    )

    # (nx+1, ny+1, nz)
    field = draw(
        st_raw_field(
            storage_shape or (nx + 1, ny + 1, nz),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["afield_at_uv_locations"] = get_dataarray_3d(
        field,
        grid,
        "J K^-1",
        grid_shape=(nx + 1, ny + 1, nz),
        set_coordinates=False,
    )

    # (nx, ny, nz+1)
    field = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz + 1),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["afield_on_interface_levels"] = get_dataarray_3d(
        field,
        grid,
        "kg m s^-2",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=False,
    )

    # (nx+1, ny, nz+1)
    field = draw(
        st_raw_field(
            storage_shape or (nx + 1, ny, nz + 1),
            -1e6,
            1e6,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict[
        "afield_at_u_locations_on_interface_levels"
    ] = get_dataarray_3d(
        field,
        grid,
        "kg m s^-2",
        grid_shape=(nx + 1, ny, nz + 1),
        set_coordinates=False,
    )

    return return_dict


@hyp_st.composite
def st_isentropic_state(
    draw,
    grid,
    *,
    time=None,
    moist=False,
    precipitation=False,
    backend="numpy",
    default_origin=None,
    storage_shape=None
):
    """ Strategy drawing a valid isentropic_prognostic model state over `grid`. """
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()
    dtype = grid.x.dtype
    set_coordinates = False
    if is_gt(backend) and storage_shape is None:
        storage_shape = (nx + 1, ny + 1, nz + 1)
    elif storage_shape is not None:
        storage_shape = (
            max(nx + 1, storage_shape[0]),
            max(ny + 1, storage_shape[1]),
            max(nz + 1, storage_shape[2]),
        )

    return_dict = {}

    # time
    if time is None:
        time = draw(hyp_st.datetimes())
    return_dict["time"] = time

    # air isentropic_prognostic density
    return_dict["air_isentropic_density"] = draw(
        st_field(
            grid,
            "isentropic_state",
            "air_isentropic_density",
            storage_shape=storage_shape or (nx, ny, nz),
            grid_shape=(nx, ny, nz),
            backend=backend,
            default_origin=default_origin,
            set_coordinates=set_coordinates,
        )
    )

    # x-velocity
    return_dict["x_velocity_at_u_locations"] = draw(
        st_field(
            grid,
            "isentropic_state",
            "x_velocity_at_u_locations",
            storage_shape=storage_shape or (nx + 1, ny, nz),
            grid_shape=(nx + 1, ny, nz),
            backend=backend,
            default_origin=default_origin,
            set_coordinates=set_coordinates,
        )
    )

    # x-momentum
    s = return_dict["air_isentropic_density"]
    u = return_dict["x_velocity_at_u_locations"]
    s_units = s.attrs["units"]
    u_units = u.attrs["units"]
    su_units = clean_units(s_units + u_units)
    su_raw = zeros(
        storage_shape or (nx, ny, nz),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    su_raw[:nx, :ny, :nz] = (
        s.to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
        * 0.5
        * (
            u.to_units("m s^-1").values[:nx, :ny, :nz]
            + u.to_units("m s^-1").values[1 : nx + 1, :ny, :nz]
        )
    )
    return_dict["x_momentum_isentropic"] = get_dataarray_3d(
        su_raw,
        grid,
        "kg m^-1 K^-1 s^-1",
        name="x_momentum_isentropic",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    ).to_units(su_units)

    # y-velocity
    return_dict["y_velocity_at_v_locations"] = draw(
        st_field(
            grid,
            "isentropic_state",
            "y_velocity_at_v_locations",
            storage_shape=storage_shape or (nx, ny + 1, nz),
            grid_shape=(nx, ny + 1, nz),
            backend=backend,
            default_origin=default_origin,
            set_coordinates=set_coordinates,
        )
    )

    # y-momentum
    v = return_dict["y_velocity_at_v_locations"]
    v_units = v.attrs["units"]
    sv_units = clean_units(s_units + v_units)
    sv_raw = zeros(
        storage_shape or (nx, ny, nz),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sv_raw[:nx, :ny, :nz] = (
        s.to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
        * 0.5
        * (
            v.to_units("m s^-1").values[:nx, :ny, :nz]
            + v.to_units("m s^-1").values[:nx, 1 : ny + 1, :nz]
        )
    )
    return_dict["y_momentum_isentropic"] = get_dataarray_3d(
        sv_raw,
        grid,
        "kg m^-1 K^-1 s^-1",
        name="y_momentum_isentropic",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    ).to_units(sv_units)

    # physical constants
    pcs = get_physical_constants(default_physical_constants)
    Rd = pcs["gas_constant_of_dry_air"]
    g = pcs["gravitational_acceleration"]
    pref = pcs["reference_air_pressure"]
    cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    # air pressure
    p = zeros(
        storage_shape or (nx, ny, nz + 1),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    p[:, :, 0] = 20
    for k in range(0, nz):
        p[:, :, k + 1] = (
            p[:, :, k] + g * dz * s.to_units("kg m^-2 K^-1").values[:, :, k]
        )
    return_dict["air_pressure_on_interface_levels"] = get_dataarray_3d(
        p,
        grid,
        "Pa",
        name="air_pressure_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )
    p[np.where(p <= 0)] = 1.0

    # exner function
    exn = zeros(
        storage_shape or (nx, ny, nz + 1),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    exn[...] = cp * (p / pref) ** (Rd / cp)
    return_dict["exner_function_on_interface_levels"] = get_dataarray_3d(
        exn,
        grid,
        "J kg^-1 K^-1",
        name="exner_function_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )

    # montgomery potential
    mtg = zeros(
        storage_shape or (nx, ny, nz),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    mtg_s = (
        grid.z_on_interface_levels.to_units("K").values[-1] * exn[:nx, :ny, nz]
        + g * grid.topography.profile.to_units("m").values[:nx, :ny]
    )
    mtg[:nx, :ny, nz - 1] = mtg_s + 0.5 * dz * exn[:nx, :ny, nz]
    for k in range(nz - 1, 0, -1):
        mtg[:, :, k - 1] = mtg[:, :, k] + dz * exn[:, :, k]
    return_dict["montgomery_potential"] = get_dataarray_3d(
        mtg,
        grid,
        "m^2 s^-2",
        name="montgomery_potential",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    )

    # height
    theta1d = grid.z_on_interface_levels.to_units("K").values
    theta = np.tile(theta1d[np.newaxis, np.newaxis, :], (nx, ny, 1))
    h = zeros(
        storage_shape or (nx, ny, nz + 1),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    h[:nx, :ny, nz] = grid.topography.profile.to_units("m").values[:nx, :ny]
    for k in range(nz, 0, -1):
        h[:nx, :ny, k - 1] = h[:nx, :ny, k] - Rd * (
            theta[:, :, k - 1] * exn[:nx, :ny, k - 1]
            + theta[:, :, k] * exn[:nx, :ny, k]
        ) * (p[:nx, :ny, k - 1] - p[:nx, :ny, k]) / (
            cp * g * (p[:nx, :ny, k - 1] + p[:nx, :ny, k])
        )
    return_dict["height_on_interface_levels"] = get_dataarray_3d(
        h,
        grid,
        "m",
        name="height_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )

    if moist:
        # air density
        rho = zeros(
            storage_shape or (nx, ny, nz),
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        rho[:nx, :ny, :nz] = (
            s.to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
            * (theta[:, :, :-1] - theta[:, :, 1:])
            / (h[:nx, :ny, :nz] - h[:nx, :ny, 1 : nz + 1])
        )
        return_dict["air_density"] = get_dataarray_3d(
            rho,
            grid,
            "kg m^-3",
            name="air_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # air temperature
        temp = zeros(
            storage_shape or (nx, ny, nz),
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        temp[:nx, :ny, :nz] = (
            0.5
            * (
                theta[:, :, 1:] * exn[:nx, :ny, 1 : nz + 1]
                + theta[:, :, -1:] * exn[:nx, :ny, :nz]
            )
            / cp
        )
        return_dict["air_temperature"] = get_dataarray_3d(
            temp,
            grid,
            "K",
            name="air_temperature",
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # mass fraction of water vapor
        return_dict["mass_fraction_of_water_vapor_in_air"] = draw(
            st_field(
                grid,
                "isentropic_state",
                "mass_fraction_of_water_vapor_in_air",
                storage_shape=storage_shape or (nx, ny, nz),
                grid_shape=(nx, ny, nz),
                backend=backend,
                default_origin=default_origin,
                set_coordinates=set_coordinates,
            )
        )

        # mass fraction of cloud liquid water
        return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = draw(
            st_field(
                grid,
                "isentropic_state",
                "mass_fraction_of_cloud_liquid_water_in_air",
                storage_shape=storage_shape or (nx, ny, nz),
                grid_shape=(nx, ny, nz),
                backend=backend,
                default_origin=default_origin,
                set_coordinates=set_coordinates,
            )
        )

        # mass fraction of precipitation water
        return_dict["mass_fraction_of_precipitation_water_in_air"] = draw(
            st_field(
                grid,
                "isentropic_state",
                "mass_fraction_of_precipitation_water_in_air",
                storage_shape=storage_shape or (nx, ny, nz),
                grid_shape=(nx, ny, nz),
                backend=backend,
                default_origin=default_origin,
                set_coordinates=set_coordinates,
            )
        )

        if precipitation:
            # precipitation
            return_dict["precipitation"] = draw(
                st_field(
                    grid,
                    "isentropic_state",
                    "precipitation",
                    storage_shape=(nx, ny, 1)
                    if storage_shape is None
                    else (storage_shape[0], storage_shape[1], 1),
                    grid_shape=(nx, ny, 1),
                    backend=backend,
                    default_origin=default_origin,
                    set_coordinates=set_coordinates,
                )
            )

            # accumulated precipitation
            return_dict["accumulated_precipitation"] = draw(
                st_field(
                    grid,
                    "isentropic_state",
                    "accumulated_precipitation",
                    storage_shape=(nx, ny, 1)
                    if storage_shape is None
                    else (storage_shape[0], storage_shape[1], 1),
                    grid_shape=(nx, ny, 1),
                    backend=backend,
                    default_origin=default_origin,
                    set_coordinates=set_coordinates,
                )
            )

    return return_dict


@hyp_st.composite
def st_isentropic_state_f(
    draw,
    grid,
    *,
    time=None,
    moist=False,
    precipitation=False,
    backend="numpy",
    default_origin=None,
    storage_shape=None
):
    """ Strategy drawing a valid isentropic_prognostic model state over `grid`. """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.x.dtype
    set_coordinates = False
    if is_gt(backend) and storage_shape is None:
        storage_shape = (nx + 1, ny + 1, nz + 1)
    elif storage_shape is not None:
        storage_shape = (
            max(nx + 1, storage_shape[0]),
            max(ny + 1, storage_shape[1]),
            max(nz + 1, storage_shape[2]),
        )

    return_dict = {}

    # time
    if time is None:
        time = draw(hyp_st.datetimes())
    return_dict["time"] = time

    # air isentropic_prognostic density
    s = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz),
            1e-1,
            1000.0,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    units = draw(
        st_one_of(conf.isentropic_state["air_isentropic_density"].keys())
    )
    return_dict["air_isentropic_density"] = get_dataarray_3d(
        s,
        grid,
        units,
        name="air_isentropic_density",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    )

    # x-velocity
    u = draw(
        st_raw_field(
            storage_shape or (nx + 1, ny, nz),
            -1000,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    units = draw(
        st_one_of(conf.isentropic_state["x_velocity_at_u_locations"].keys())
    )
    return_dict["x_velocity_at_u_locations"] = get_dataarray_3d(
        u,
        grid,
        units,
        name="x_velocity_at_u_locations",
        grid_shape=(nx + 1, ny, nz),
        set_coordinates=set_coordinates,
    )

    # x-momentum
    s = return_dict["air_isentropic_density"]
    u = return_dict["x_velocity_at_u_locations"]
    s_units = s.attrs["units"]
    u_units = u.attrs["units"]
    su_units = clean_units(s_units + u_units)
    su_raw = zeros(
        storage_shape or (nx, ny, nz),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    su_raw[:nx, :ny, :nz] = (
        s.to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
        * 0.5
        * (
            u.to_units("m s^-1").values[:nx, :ny, :nz]
            + u.to_units("m s^-1").values[1 : nx + 1, :ny, :nz]
        )
    )
    return_dict["x_momentum_isentropic"] = get_dataarray_3d(
        su_raw,
        grid,
        "kg m^-1 K^-1 s^-1",
        name="x_momentum_isentropic",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    ).to_units(su_units)

    # y-velocity
    v = draw(
        st_raw_field(
            storage_shape or (nx, ny + 1, nz),
            -1000,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    units = draw(
        st_one_of(conf.isentropic_state["y_velocity_at_v_locations"].keys())
    )
    return_dict["y_velocity_at_v_locations"] = get_dataarray_3d(
        v,
        grid,
        units,
        name="y_velocity_at_v_locations",
        grid_shape=(nx, ny + 1, nz),
        set_coordinates=set_coordinates,
    )

    # y-momentum
    v = return_dict["y_velocity_at_v_locations"]
    v_units = v.attrs["units"]
    sv_units = clean_units(s_units + v_units)
    sv_raw = zeros(
        storage_shape or (nx, ny, nz),
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sv_raw[:nx, :ny, :nz] = (
        s.to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
        * 0.5
        * (
            v.to_units("m s^-1").values[:nx, :ny, :nz]
            + v.to_units("m s^-1").values[:nx, 1 : ny + 1, :nz]
        )
    )
    return_dict["y_momentum_isentropic"] = get_dataarray_3d(
        sv_raw,
        grid,
        "kg m^-1 K^-1 s^-1",
        name="y_momentum_isentropic",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    ).to_units(sv_units)

    # air pressure
    p = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz + 1),
            1e-8,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["air_pressure_on_interface_levels"] = get_dataarray_3d(
        p,
        grid,
        "Pa",
        name="air_pressure_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )

    # exner function
    exn = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz + 1),
            1e-8,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["exner_function_on_interface_levels"] = get_dataarray_3d(
        exn,
        grid,
        "J kg^-1 K^-1",
        name="exner_function_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )

    # montgomery potential
    mtg = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz),
            1e-8,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["montgomery_potential"] = get_dataarray_3d(
        mtg,
        grid,
        "m^2 s^-2",
        name="montgomery_potential",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    )

    # height
    h = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz + 1),
            1e-8,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["height_on_interface_levels"] = get_dataarray_3d(
        h,
        grid,
        "m",
        name="height_on_interface_levels",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=set_coordinates,
    )

    if moist:
        # air density
        rho = draw(
            st_raw_field(
                storage_shape or (nx, ny, nz),
                1e-8,
                1000,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
            )
        )
        return_dict["air_density"] = get_dataarray_3d(
            rho,
            grid,
            "kg m^-3",
            name="air_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # air temperature
        t = draw(
            st_raw_field(
                storage_shape or (nx, ny, nz),
                1e-8,
                1000,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
            )
        )
        return_dict["air_temperature"] = get_dataarray_3d(
            t,
            grid,
            "K",
            name="air_temperature",
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # mass fraction of water vapor
        q = draw(
            st_raw_field(
                storage_shape or (nx, ny, nz),
                0,
                1000,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
            )
        )
        units = draw(st_one_of(conf.isentropic_state[mfwv].keys()))
        return_dict[mfwv] = get_dataarray_3d(
            q,
            grid,
            units,
            name=mfwv,
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # mass fraction of cloud liquid water
        q = draw(
            st_raw_field(
                storage_shape or (nx, ny, nz),
                0,
                1000,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
            )
        )
        units = draw(st_one_of(conf.isentropic_state[mfcw].keys()))
        return_dict[mfcw] = get_dataarray_3d(
            q,
            grid,
            units,
            name=mfcw,
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        # mass fraction of precipitation water
        q = draw(
            st_raw_field(
                storage_shape or (nx, ny, nz),
                0,
                1000,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
            )
        )
        units = draw(st_one_of(conf.isentropic_state[mfpw].keys()))
        return_dict[mfpw] = get_dataarray_3d(
            q,
            grid,
            units,
            name=mfpw,
            grid_shape=(nx, ny, nz),
            set_coordinates=set_coordinates,
        )

        if precipitation:
            # precipitation
            pp = draw(
                st_raw_field(
                    (nx, ny, 1)
                    if storage_shape is None
                    else (storage_shape[0], storage_shape[1], 1),
                    0,
                    1000,
                    backend=backend,
                    dtype=dtype,
                    default_origin=default_origin,
                )
            )
            units = draw(
                st_one_of(conf.isentropic_state["precipitation"].keys())
            )
            return_dict["precipitation"] = get_dataarray_3d(
                pp,
                grid,
                units,
                name="precipitation",
                grid_shape=(nx, ny, 1),
                set_coordinates=set_coordinates,
            )

            # accumulated precipitation
            app = draw(
                st_raw_field(
                    (nx, ny, 1)
                    if storage_shape is None
                    else (storage_shape[0], storage_shape[1], 1),
                    0,
                    1000,
                    backend=backend,
                    dtype=dtype,
                    default_origin=default_origin,
                )
            )
            units = draw(
                st_one_of(
                    conf.isentropic_state["accumulated_precipitation"].keys()
                )
            )
            return_dict["accumulated_precipitation"] = get_dataarray_3d(
                app,
                grid,
                units,
                name="accumulated_precipitation",
                grid_shape=(nx, ny, 1),
                set_coordinates=set_coordinates,
            )

    return return_dict


@hyp_st.composite
def st_isentropic_boussinesq_state_f(
    draw,
    grid,
    *,
    time=None,
    moist=False,
    precipitation=False,
    backend="numpy",
    default_origin=None,
    storage_shape=None
):
    """ Strategy drawing a valid isentropic_prognostic Boussinesq model state over `grid`. """
    return_dict = draw(
        st_isentropic_state_f(
            grid,
            time=time,
            moist=moist,
            precipitation=precipitation,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        )
    )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.x.dtype
    set_coordinates = storage_shape is not None
    if storage_shape is not None:
        storage_shape[0] = max(nx + 1, storage_shape[0])
        storage_shape[1] = max(ny + 1, storage_shape[1])
        storage_shape[2] = max(nz + 1, storage_shape[2])

    ddmtg = draw(
        st_raw_field(
            storage_shape or (nx, ny, nz),
            0,
            1000,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    return_dict["dd_montgomery_potential"] = get_dataarray_3d(
        ddmtg,
        grid,
        "m^2 K^-2 s^-2",
        name="dd_montgomery_potential",
        grid_shape=(nx, ny, nz),
        set_coordinates=set_coordinates,
    )

    return return_dict


@hyp_st.composite
def st_burgers_state(
    draw, grid, *, time=None, backend="numpy", default_origin=None
):
    """ Strategy drawing a valid Burgers model state over `grid`. """
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    assert nz == 1

    return_dict = {}

    # time
    if time is None:
        time = draw(hyp_st.datetimes())
    return_dict["time"] = time

    # x-velocity
    return_dict["x_velocity"] = draw(
        st_field(
            grid,
            "burgers_state",
            "x_velocity",
            backend=backend,
            default_origin=default_origin,
            set_coordinates=True,
        )
    )

    # y-velocity
    return_dict["y_velocity"] = draw(
        st_field(
            grid,
            "burgers_state",
            "y_velocity",
            backend=backend,
            default_origin=default_origin,
            set_coordinates=True,
        )
    )

    return return_dict


@hyp_st.composite
def st_burgers_tendency(
    draw, grid, *, time=None, backend="numpy", default_origin=None
):
    """
    Strategy drawing a set of tendencies for the variables whose evolution is
    governed by the Burgers equations.
    """
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    assert nz == 1

    return_dict = {}

    # time
    if time is None:
        time = draw(hyp_st.datetimes())
    return_dict["time"] = time

    # x-velocity
    return_dict["x_velocity"] = draw(
        st_field(
            grid,
            "burgers_tendency",
            "x_velocity",
            backend=backend,
            default_origin=default_origin,
            set_coordinates=True,
        )
    )

    # y-velocity
    return_dict["y_velocity"] = draw(
        st_field(
            grid,
            "burgers_tendency",
            "y_velocity",
            backend=backend,
            default_origin=default_origin,
            set_coordinates=True,
        )
    )

    return return_dict
