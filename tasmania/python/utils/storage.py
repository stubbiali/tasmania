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
from copy import deepcopy
import numpy as np
from sympl import DataArray
from typing import TYPE_CHECKING, Optional, Sequence, Union

try:
    import cupy as cp
except ImportError:
    cp = np

from tasmania.python.framework.allocators import as_storage
from tasmania.python.utils import typingx as ty

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.domain.grid import Grid, NumericalGrid, PhysicalGrid
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
    from tasmania.python.domain.horizontal_grid import HorizontalGrid
    from tasmania.python.framework.options import StorageOptions


def get_dataarray_2d(
    array: ty.Storage,
    grid: "Union[Grid, HorizontalGrid]",
    units: str,
    name: Optional[str] = None,
    grid_origin: Optional[ty.PairInt] = None,
    grid_shape: Optional[ty.PairInt] = None,
    set_coordinates: bool = True,
) -> DataArray:
    """Create a DataArray out of a 2-D :class:`numpy.ndarray`-like storage.

    Parameters
    ----------
    array : array-like
        2-D buffer storing the field values.
    grid : tasmania.Grid
        The underlying grid.
    units : str
        The variable units.
    name : `str`, optional
        The variable name. Defaults to `None`.
    grid_origin : `Sequence[int]`, optional
        The index of the element in the buffer associated with the (0, 0)
        grid point. If not specified, it is assumed that `grid_origin = (0, 0)`.
    grid_shape : `Sequence[int]`, optional
        The shape of grid underlying the field. It cannot exceed the shape
        of the passed buffer. If not specified, it is assumed that it coincides
        with the shape of the buffer.
    set_coordinates : `bool`, optional
        ``True`` to set the coordinates of the grid points, ``False`` otherwise.

    Return
    ------
    sympl.DataArray :
        The :class:`sympl.DataArray` whose value array is `array`,
        whose coordinates and dimensions are retrieved from `grid`,
        and whose units are `units`.
    """
    nx, ny = grid.nx, grid.ny
    grid_origin = (0, 0) if grid_origin is None else grid_origin
    grid_shape = array.shape if grid_shape is None else grid_shape
    try:
        ni, nj = grid_shape
    except ValueError:
        raise ValueError(
            f"Expected a 2-D array, got a {len(grid_shape)}-D one."
        )

    if ni == nx:
        x = grid.x
    elif ni == nx + 1:
        x = grid.x_at_u_locations
    else:
        raise ValueError(
            f"The array extent in the x-direction is {ni} but either "
            f"{nx} or {nx+1} was expected."
        )

    if nj == ny:
        y = grid.y
    elif nj == ny + 1:
        y = grid.y_at_v_locations
    else:
        raise ValueError(
            f"The array extent in the y-direction is {nj} but either "
            f"{ny} or {ny+1} was expected."
        )

    if set_coordinates:
        xslice = slice(grid_origin[0], grid_origin[0] + ni)
        yslice = slice(grid_origin[1], grid_origin[1] + nj)
        return DataArray(
            array[xslice, yslice],
            coords=[x.coords[x.dims[0]].values, y.coords[y.dims[0]].values],
            dims=[x.dims[0], y.dims[0]],
            name=name,
            attrs={"units": units},
        )
    else:
        return DataArray(
            array,
            dims=[x.dims[0], y.dims[0]],
            name=name,
            attrs={"units": units},
        )


def get_dataarray_3d(
    array: ty.Storage,
    grid: "Grid",
    units: str,
    name: Optional[str] = None,
    grid_origin: Optional[ty.TripletInt] = None,
    grid_shape: Optional[ty.TripletInt] = None,
    set_coordinates: bool = True,
) -> DataArray:
    """Create a DataArray out of a 3-D ndarray-like storage.

    Parameters
    ----------
    array : array-like
        3-D buffer storing the field values.
    grid : tasmania.Grid
        The underlying grid.
    units : str
        The variable units.
    name : `str`, optional
        The variable name. Defaults to `None`.
    grid_origin : `Sequence[int]`, optional
        The index of the element in the buffer associated with the (0, 0, 0)
        grid point. If not specified, it is assumed that
        `grid_origin = (0, 0, 0)`.
    grid_shape : `Sequence[int]`, optional
        The shape of grid underlying the field. It cannot exceed the shape
        of the passed buffer. If not specified, it is assumed that it coincides
        with the shape of the buffer.
    set_coordinates : `bool`, optional
        ``True`` to set the coordinates of the grid points,
        ``False`` otherwise.

    Return
    ------
    sympl.DataArray :
        The :class:`sympl.DataArray` whose value array is `array`,
        whose coordinates and dimensions are retrieved from `grid`,
        and whose units are `units`.
    """
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    grid_origin = grid_origin or (0, 0, 0)
    grid_shape = grid_shape or array.shape
    try:
        ni, nj, nk = grid_shape
    except ValueError:
        raise ValueError(
            f"Expected a 3-D array, got a {len(grid_shape)}-D one."
        )

    if ni == 1 and nx != 1:
        x = DataArray(
            np.array([grid.grid_xy.x.values[0]]),
            dims=grid.grid_xy.x.dims[0] + "_gp",
            attrs={"units": grid.grid_xy.x.attrs["units"]},
        )
    elif ni == nx:
        x = grid.grid_xy.x
    elif ni == nx + 1:
        x = grid.grid_xy.x_at_u_locations
    else:
        raise ValueError(
            f"The grid extent in the x-direction is {ni} but either "
            f"{1}, {nx} or {nx+1} was expected."
        )

    if nj == 1 and ny != 1:
        y = DataArray(
            np.array([grid.grid_xy.y.values[0]]),
            dims=grid.grid_xy.y.dims[0] + "_gp",
            attrs={"units": grid.grid_xy.y.attrs["units"]},
        )
    elif nj == ny:
        y = grid.grid_xy.y
    elif nj == ny + 1:
        y = grid.grid_xy.y_at_v_locations
    else:
        raise ValueError(
            f"The grid extent in the y-direction is {nj} but either "
            f"1, {ny} or {ny+1} was expected."
        )

    if nk == 1:
        if nz > 1:
            z = DataArray(
                np.array([grid.z_on_interface_levels.values[-1]]),
                dims=grid.z.dims[0] + "_at_surface_level",
                attrs={"units": grid.z.attrs["units"]},
            )
        else:
            z = DataArray(
                np.array([grid.z.values[-1]]),
                dims=grid.z.dims[0],
                attrs={"units": grid.z.attrs["units"]},
            )
    elif nk == nz:
        z = grid.z
    elif nk == nz + 1:
        z = grid.z_on_interface_levels
    else:
        raise ValueError(
            f"The grid extent in the z-direction is {nk} but either "
            f"1, {nz} or {nz+1} was expected."
        )

    if set_coordinates:
        xslice = slice(grid_origin[0], grid_origin[0] + ni)
        yslice = slice(grid_origin[1], grid_origin[1] + nj)
        zslice = slice(grid_origin[2], grid_origin[2] + nk)
        return DataArray(
            array[xslice, yslice, zslice],
            coords=[
                x.coords[x.dims[0]].values,
                y.coords[y.dims[0]].values,
                z.coords[z.dims[0]].values,
            ],
            dims=[x.dims[0], y.dims[0], z.dims[0]],
            name=name,
            attrs={"units": units},
        )
    else:
        return DataArray(
            array,
            dims=[x.dims[0], y.dims[0], z.dims[0]],
            name=name,
            attrs={"units": units},
        )


def get_dataarray_dict(
    array_dict: ty.StorageDict, grid: "Grid", properties: ty.PropertiesDict
) -> ty.DataArrayDict:
    """
    Parameters
    ----------
    array_dict[str, array_like] dict
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are :class:`numpy.ndarray`-like
        arrays containing the data for those variables.
    grid : tasmania.Grid
        The underlying grid.
    properties : dict[str, str]
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are strings indicating
        the units in which those variables should be expressed.

    Return
    ------
    dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are :class:`sympl.DataArray`\s
        containing the data for those variables.
    """
    try:
        dataarray_dict = {"time": array_dict["time"]}
    except KeyError:
        dataarray_dict = {}

    for key in array_dict.keys():
        if key != "time":
            units = properties[key]["units"]
            grid_origin = properties[key].get("grid_origin", None)
            grid_shape = properties[key].get("grid_shape", None)
            set_coordinates = properties[key].get("set_coordinates", True)
            if len(array_dict[key].shape) == 2:
                dataarray_dict[key] = get_dataarray_2d(
                    array_dict[key],
                    grid,
                    units,
                    name=key,
                    grid_origin=grid_origin,
                    grid_shape=grid_shape,
                    set_coordinates=set_coordinates,
                )
            else:
                dataarray_dict[key] = get_dataarray_3d(
                    array_dict[key],
                    grid,
                    units,
                    name=key,
                    grid_origin=grid_origin,
                    grid_shape=grid_shape,
                    set_coordinates=set_coordinates,
                )

    return dataarray_dict


def get_array_dict(
    dataarray_dict: ty.DataArrayDict, properties: ty.PropertiesDict
) -> ty.StorageDict:
    """
    Parameters
    ----------
    dataarray_dict : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    properties : dict[str, dict]
        Dictionary whose keys are strings indicating the variable names in
        `dataarray_dict`, and values are dictionaries storing fundamental
        properties (units) for those variables.

    Return
    ------
    dict[str, array_like] :
        Dictionary whose keys are strings indicating the variable names in
        `dataarray_dict`,  and values are :class:`numpy.ndarray`-like arrays
        containing the data for those variables.
    """
    try:
        array_dict = {"time": dataarray_dict["time"]}
    except KeyError:
        array_dict = {}

    for key in dataarray_dict.keys():
        if key != "time":
            props = properties.get(key, {})
            units = props.get("units", dataarray_dict[key].attrs.get("units"))
            assert units is not None, "Units not specified for {}.".format(key)
            array_dict[key] = dataarray_dict[key].to_units(units).data

    return array_dict


def get_physical_state(
    pgrid: "PhysicalGrid",
    hb: "HorizontalBoundary",
    cstate: ty.DataArrayDict,
    store_names: Optional[Sequence[str]] = None,
) -> ty.DataArrayDict:
    """
    Given a state dictionary defined over the numerical grid, transpose
    that state over the corresponding physical grid.
    """
    nx, ny, nz = pgrid.nx, pgrid.ny, pgrid.nz

    store_names = (
        store_names
        if store_names is not None
        else tuple(name for name in cstate if name != "time")
    )
    store_names = tuple(name for name in store_names if name in cstate)

    pstate = {"time": cstate["time"]} if "time" in cstate else {}

    for name in store_names:
        if name != "time":
            storage_shape = cstate[name].shape
            mx = nx + 1 if "at_u_locations" in name else nx
            my = ny + 1 if "at_v_locations" in name else ny
            mz = nz + 1 if "on_interface_levels" in name else nz
            units = cstate[name].attrs["units"]

            raw_cfield = cstate[name].data
            raw_pfield = hb.get_physical_field(raw_cfield, name)

            if len(storage_shape) == 2:
                pstate[name] = get_dataarray_2d(
                    raw_pfield[:mx, :my],
                    pgrid,
                    units,
                    name=name,
                    set_coordinates=True,
                )
            else:
                pstate[name] = get_dataarray_3d(
                    raw_pfield[:mx, :my, :mz],
                    pgrid,
                    units,
                    name=name,
                    set_coordinates=True,
                )

    return pstate


def get_numerical_state(
    ngrid: "NumericalGrid",
    hb: "HorizontalBoundary",
    pstate: ty.DataArrayDict,
    store_names: Optional[Sequence[str]] = None,
) -> ty.DataArrayDict:
    """
    Given a state defined over the physical grid, transpose that state
    over the corresponding numerical grid.
    """
    nx, ny, nz = ngrid.nx, ngrid.ny, ngrid.nz

    store_names = (
        store_names
        if store_names is not None
        else tuple(name for name in pstate if name != "time")
    )
    store_names = tuple(name for name in store_names if name in pstate)

    nstate = {"time": pstate["time"]} if "time" in pstate else {}

    for name in store_names:
        if name != "time":
            mx = nx + 1 if "at_u_locations" in name else nx
            my = ny + 1 if "at_v_locations" in name else ny
            mz = nz + 1 if "on_interface_levels" in name else nz
            units = pstate[name].attrs["units"]

            raw_pfield = pstate[name].data
            raw_nfield = hb.get_numerical_field(raw_pfield, name)

            if raw_nfield.ndim == 2:
                nstate[name] = get_dataarray_2d(
                    raw_nfield,
                    ngrid,
                    units,
                    name,
                    grid_shape=(mx, my),
                    set_coordinates=True,
                )
            else:
                nstate[name] = get_dataarray_3d(
                    raw_nfield,
                    ngrid,
                    units,
                    name,
                    grid_shape=(mx, my, mz),
                    set_coordinates=True,
                )

    return nstate


def get_min_storage_shape(name, grid):
    out = (
        grid.nx + 1
        if "at_u_locations" in name or "at_uv_locations" in name
        else grid.nx,
        grid.ny + 1
        if "at_v_locations" in name or "at_uv_locations" in name
        else grid.ny,
        grid.nz + 1 if "on_interface_levels" in name else grid.nz,
    )
    return out


def get_storage_shape(
    in_shape: Sequence[int],
    min_shape: Sequence[int],
    max_shape: Optional[Sequence[int]] = None,
) -> Sequence[int]:
    out_shape = in_shape or min_shape

    if max_shape is None:
        error_msg = "storage shape must be larger or equal than {}.".format(
            min_shape
        )
        assert all(
            tuple(out_shape[i] >= min_shape[i] for i in range(len(min_shape)))
        ), error_msg
    else:
        error_msg = "storage shape must be between {} and {}".format(
            min_shape, max_shape
        )
        assert all(
            tuple(
                min_shape[i] <= out_shape[i] <= max_shape[i]
                for i in range(len(min_shape))
            )
        ), error_msg

    return out_shape


def get_aligned_index(
    aligned_index: ty.TripletInt,
    storage_shape: ty.TripletInt,
    min_aligned_index: Optional[ty.TripletInt] = None,
    max_aligned_index: Optional[ty.TripletInt] = None,
) -> ty.TripletInt:
    aligned_index = aligned_index or (0, 0, 0)

    max_aligned_index = max_aligned_index or aligned_index
    max_aligned_index = tuple(
        max_aligned_index[i]
        if storage_shape[i] > 2 * max_aligned_index[i]
        else 0
        for i in range(3)
    )

    min_aligned_index = min_aligned_index or max_aligned_index
    min_aligned_index = tuple(
        min_aligned_index[i]
        if min_aligned_index[i] <= max_aligned_index[i]
        else max_aligned_index[i]
        for i in range(3)
    )

    out = tuple(
        aligned_index[i]
        if min_aligned_index[i] <= aligned_index[i] <= max_aligned_index[i]
        else min_aligned_index[i]
        for i in range(3)
    )

    return out


def deepcopy_array_dict(
    src: ty.StorageDict,
    *,
    backend: Optional[str] = None,
    storage_options: Optional["StorageOptions"] = None
) -> ty.StorageDict:
    dst = {"time": src["time"]} if "time" in src else {}
    for name in src:
        if name != "time":
            if backend is not None:
                dst[name] = as_storage(
                    backend, data=src[name], storage_options=storage_options
                )
            else:
                dst[name] = deepcopy(src[name])
    return dst


def deepcopy_dataarray(
    src: DataArray,
    *,
    backend: Optional[str] = None,
    storage_options: Optional["StorageOptions"] = None
) -> DataArray:
    raw_array = (
        as_storage(backend, data=src.data, storage_options=storage_options)
        if backend is not None
        else deepcopy(src.data)
    )
    return DataArray(
        raw_array,
        coords=src.coords,
        dims=src.dims,
        name=src.name,
        attrs=src.attrs.copy(),
    )


def deepcopy_dataarray_dict(
    src: ty.DataArrayDict,
    *,
    backend: Optional[str] = None,
    storage_options: Optional["StorageOptions"] = None
) -> ty.DataArrayDict:
    dst = {"time": src["time"]} if "time" in src else {}
    for name in src:
        if name != "time":
            dst[name] = deepcopy_dataarray(
                src[name], backend=backend, storage_options=storage_options
            )
    return dst
