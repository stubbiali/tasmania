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

import gridtools as gt


def get_dataarray_2d(
    array, grid, units, name=None, grid_origin=None, grid_shape=None, set_coordinates=True
):
    """
    Create a DataArray out of a 2-D ndarray-like storage.

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
    grid_origin : `sequence`, optional
        The index of the element in the buffer associated with the (0, 0)
        grid point. If not specified, it is assumed that `grid_origin = (0, 0)`.
    grid_shape : `sequence`, optional
        The shape of grid underlying the field. It cannot exceed the shape
        of the passed buffer. If not specified, it is assumed that it coincides
        with the shape of the buffer.
    set_coordinates : `bool`, optional
        TODO

    Return
    ------
    dataarray-like :
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
        raise ValueError("Expected a 2-D array, got a {}-D one.".format(len(grid_shape)))

    if ni == nx:
        x = grid.x
    elif ni == nx + 1:
        x = grid.x_at_u_locations
    else:
        raise ValueError(
            "The array extent in the x-direction is {} but either "
            "{} or {} was expected.".format(ni, nx, nx + 1)
        )

    if nj == ny:
        y = grid.y
    elif nj == ny + 1:
        y = grid.y_at_v_locations
    else:
        raise ValueError(
            "The array extent in the y-direction is {} but either "
            "{} or {} was expected.".format(nj, ny, ny + 1)
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
            array, dims=[x.dims[0], y.dims[0]], name=name, attrs={"units": units}
        )


def get_dataarray_3d(
    array, grid, units, name=None, grid_origin=None, grid_shape=None, set_coordinates=True
):
    """
    Create a DataArray out of a 3-D ndarray-like storage.

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
    grid_origin : `sequence`, optional
        The index of the element in the buffer associated with the (0, 0, 0)
        grid point. If not specified, it is assumed that `grid_origin = (0, 0, 0)`.
    grid_shape : `sequence`, optional
        The shape of grid underlying the field. It cannot exceed the shape
        of the passed buffer. If not specified, it is assumed that it coincides
        with the shape of the buffer.
    set_coordinates : `bool`, optional
        TODO

    Return
    ------
    dataarray-like :
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
        raise ValueError("Expected a 3-D array, got a {}-D one.".format(len(grid_shape)))

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
            "The grid extent in the x-direction is {} but either "
            "{}, {} or {} was expected.".format(ni, 1, nx, nx + 1)
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
            "The grid extent in the y-direction is {} but either "
            "{}, {} or {} was expected.".format(nj, 1, ny, ny + 1)
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
            "The grid extent in the z-direction is {} but either "
            "1, {} or {} was expected.".format(nk, nz, nz + 1)
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


def get_dataarray_dict(array_dict, grid, properties, set_coordinates=True):
    """
    Parameters
    ----------
    array_dict : dict
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are :class:`numpy.ndarray`\s
        containing the data for those variables.
    grid : tasmania.Grid
        The underlying grid.
    properties : dict
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are strings indicating
        the units in which those variables should be expressed.
    set_coordinates : `bool`, optional
        TODO

    Return
    ------
    dict :
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


def get_array_dict(dataarray_dict, properties):
    """
    Parameters
    ----------
    dataarray_dict : dict
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are :class:`sympl.DataArray`\s
        containing the data for those variables.
    properties : dict
        TODO

    Return
    ------
    dict :
        Dictionary whose keys are strings indicating the variables
        included in the model state, and values are :class:`numpy.ndarray`\s
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
            array_dict[key] = dataarray_dict[key].to_units(units).values

    return array_dict


def get_physical_state(domain, cstate, properties, store_names=None):
    """
    Given a state dictionary defined over the numerical grid, transpose that state
    over the corresponding physical grid.
    """
    pgrid = domain.physical_grid
    hb = domain.horizontal_boundary

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
            if len(storage_shape) == 2:
                grid_origin = properties[name].get("grid_origin", (0, 0))
                grid_shape = properties[name].get("grid_shape", storage_shape)
                xslice = slice(grid_origin[0], grid_origin[0] + grid_shape[0])
                yslice = slice(grid_origin[1], grid_origin[1] + grid_shape[1])
                units = cstate[name].attrs["units"]
                raw_cfield = cstate[name].values[xslice, yslice]
                raw_pfield = hb.get_physical_field(raw_cfield, name)
                pstate[name] = get_dataarray_2d(
                    raw_pfield, pgrid, units, name=name, set_coordinates=True
                )
            else:
                grid_origin = properties[name].get("grid_origin", (0, 0, 0))
                grid_shape = properties[name].get("grid_shape", storage_shape)
                xslice = slice(grid_origin[0], grid_origin[0] + grid_shape[0])
                yslice = slice(grid_origin[1], grid_origin[1] + grid_shape[1])
                zslice = slice(grid_origin[2], grid_origin[2] + grid_shape[2])
                units = cstate[name].attrs["units"]
                raw_cfield = cstate[name].values[xslice, yslice, zslice]
                raw_pfield = hb.get_physical_field(raw_cfield, name)
                pstate[name] = get_dataarray_3d(
                    raw_pfield, pgrid, units, name=name, set_coordinates=True
                )

    return pstate


def get_numerical_state(domain, pstate, store_names=None):
    """
    Given a state defined over the physical grid, transpose that state
    over the corresponding numerical grid.
    """
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary

    store_names = (
        store_names
        if store_names is not None
        else tuple(name for name in pstate if name != "time")
    )
    store_names = tuple(name for name in store_names if name in pstate)

    cstate = {"time": pstate["time"]} if "time" in pstate else {}

    for name in store_names:
        if name != "time":
            units = pstate[name].attrs["units"]
            raw_pfield = pstate[name].values
            raw_cfield = hb.get_numerical_field(raw_pfield, name)
            if len(raw_cfield.shape) == 2:
                cstate[name] = get_dataarray_2d(
                    raw_cfield, cgrid, units, name, set_coordinates=True
                )
            else:
                cstate[name] = get_dataarray_3d(
                    raw_cfield, cgrid, units, name, set_coordinates=True
                )

    return cstate


def get_storage_descriptor(storage_shape, dtype, halo=None, mask=(True, True, True)):
    halo = (0, 0, 0) if halo is None else halo
    halo = tuple(halo[i] if storage_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(storage_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(
        dtype=dtype, mask=mask, halo=halo, iteration_domain=domain
    )
    return descriptor


def get_storage_shape(in_shape, min_shape, max_shape=None):
    out_shape = in_shape or min_shape

    if max_shape is None:
        error_msg = "storage shape must be larger or equal than {}.".format(min_shape)
        assert all(
            tuple(out_shape[i] >= min_shape[i] for i in range(len(min_shape)))
        ), error_msg
    else:
        error_msg = "storage shape must be between {} and {}".format(min_shape, max_shape)
        assert all(
            tuple(
                min_shape[i] <= out_shape[i] <= max_shape[i]
                for i in range(len(min_shape))
            )
        ), error_msg

    return out_shape


def empty(storage_shape, backend, dtype, halo=None, mask=None):
    descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo, mask=mask)
    gt_storage = gt.storage.empty(descriptor=descriptor, backend=backend)
    return gt_storage


def zeros(storage_shape, backend, dtype, halo=None, mask=None):
    descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo, mask=mask)
    gt_storage = gt.storage.zeros(descriptor=descriptor, backend=backend)
    return gt_storage


def ones(storage_shape, backend, dtype, halo=None, mask=None):
    descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo, mask=mask)
    gt_storage = gt.storage.ones(descriptor=descriptor, backend=backend)
    return gt_storage


def deepcopy_array_dict(src):
    dst = {'time': src['time']} if 'time' in src else {}
    for name in src:
        if name != 'time':
            dst[name] = deepcopy(src[name])
    return dst


def deepcopy_dataarray(src):
    return DataArray(
        deepcopy(src.values),
        coords=src.coords,
        dims=src.dims,
        name=src.name,
        attrs=src.attrs.copy(),
    )


def deepcopy_dataarray_dict(src):
    dst = {'time': src['time']} if 'time' in src else {}
    for name in src:
        if name != 'time':
            dst[name] = deepcopy_dataarray(src[name])
    return dst
