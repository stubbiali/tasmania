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
from datetime import timedelta
import netCDF4 as nc4
import numpy as np
from pandas import Timedelta
import sympl
import xarray as xr

from tasmania.python.burgers.state import ZhaoSolutionFactory
from tasmania.python.grids.domain import Domain
from tasmania.python.utils.storage_utils import (
    deepcopy_dataarray_dict,
    get_physical_state,
    get_numerical_state,
)
from tasmania.python.utils.utils import convert_datetime64_to_datetime


class NetCDFMonitor(sympl.NetCDFMonitor):
    """
    Customized version of :class:`sympl.NetCDFMonitor`, which
    caches stored states and then write them to a NetCDF file,
    together with some grid properties.
    """

    def __init__(
        self,
        filename,
        domain,
        grid_type,
        time_units="seconds",
        store_names=None,
        write_on_store=False,
        aliases=None,
    ):
        """
        The constructor.

        Parameters
        ----------
        filename : str
            The path where the NetCDF file will be written.
        domain : tasmania.Domain
            The underlying domain.
        grid_type : str
            String specifying the type of grid over which the states should be saved.
            Either:

                * 'physical';
                * 'numerical'.

        time_units : str, optional
            The units in which time will be
            stored in the NetCDF file. Time is stored as an integer
            number of these units. Default is seconds.
        store_names : iterable of str, optional
            Names of quantities to store. If not given,
            all quantities are stored.
        write_on_store : bool, optional
            If True, stored changes are immediately written to file.
            This can result in many file open/close operations.
            Default is to write only when the write() method is
            called directly.
        aliases : dict
            A dictionary of string replacements to apply to state variable
            names before saving them in netCDF files.
        """
        super().__init__(filename, time_units, store_names, write_on_store, aliases)
        self._domain = domain
        self._gtype = grid_type

    def store(self, state):
        """
        If the state is defined over the numerical (respectively physical)
        grid but should be saved over the physical (resp. numerical) grid:
        transpose the state over the appropriate grid, make a deep copy of this
        new state, and call the parent's method.
        If the state is already defined over the expected grid: make a deep copy
        of the input state before calling the parent's method.
        """
        grid = (
            self._domain.physical_grid
            if self._gtype == "physical"
            else self._domain.numerical_grid
        )
        dims_x = (grid.x.dims[0], grid.x_at_u_locations.dims[0])
        dims_y = (grid.y.dims[0], grid.y_at_v_locations.dims[0])

        if self._gtype == "physical":
            names = tuple(key for key in state if key != "time")
            if not (
                state[names[0]].dims[0] in dims_x and state[names[0]].dims[1] in dims_y
            ):
                to_save = get_physical_state(self._domain, state, self._store_names)
            else:
                to_save = state
        else:
            names = tuple(key for key in state if key != "time")
            if not (
                state[names[0]].dims[0] in dims_x and state[names[0]].dims[1] in dims_y
            ):
                to_save = get_numerical_state(self._domain, state, self._store_names)
            else:
                to_save = state

        to_save_cp = deepcopy_dataarray_dict(to_save)
        for name in to_save_cp:
            if name != "time":
                to_save_cp[name].attrs.pop("gt_storage", None)

        super().store(to_save_cp)

    def write(self):
        """
        Write grid properties and all cached states to the NetCDF file,
        and clear the cache. This will append to any existing NetCDF file.
        """
        super().write()

        with nc4.Dataset(self._filename, "a") as dataset:
            g = self._domain.physical_grid

            dataset.createDimension("bool_dim", 1)
            dataset.createDimension("scalar_dim", 1)
            dataset.createDimension("str_dim", 1)
            dataset.createDimension("timedelta_dim", 1)
            dataset.createDimension("functor_dim", 1)

            # list model state variable names
            names = [var for var in dataset.variables if var != "time"]
            dataset.createDimension("strvec1_dim", len(names))
            state_variable_names = dataset.createVariable(
                "state_variable_names", str, ("strvec1_dim",)
            )
            state_variable_names[:] = np.array(names, dtype="object")

            # type of the underlying grid over which the states are defined
            grid_type = dataset.createVariable("grid_type", str, ("str_dim",))
            grid_type[:] = np.array([self._gtype], dtype="object")

            # x-axis
            dim1_name = dataset.createVariable("dim1_name", str, ("str_dim",))
            dim1_name[:] = np.array([g.x.dims[0]], dtype="object")
            dim1 = dataset.createVariable(g.x.dims[0], g.x.values.dtype, (g.x.dims[0],))
            dim1[:] = g.x.values[:]
            dim1.setncattr("units", g.x.attrs["units"])
            try:
                dim1_u = dataset.createVariable(
                    g.x_at_u_locations.dims[0],
                    g.x_at_u_locations.values.dtype,
                    (g.x_at_u_locations.dims[0],),
                )
                dim1_u[:] = g.x_at_u_locations.values[:]
                dim1_u.setncattr("units", g.x_at_u_locations.attrs["units"])
            except ValueError:
                pass

            # y-axis
            dim2_name = dataset.createVariable("dim2_name", str, ("str_dim",))
            dim2_name[:] = np.array([g.y.dims[0]], dtype="object")
            dim2 = dataset.createVariable(g.y.dims[0], g.y.values.dtype, (g.y.dims[0],))
            dim2[:] = g.y.values[:]
            dim2.setncattr("units", g.y.attrs["units"])
            try:
                dim2_v = dataset.createVariable(
                    g.y_at_v_locations.dims[0],
                    g.y_at_v_locations.values.dtype,
                    (g.y_at_v_locations.dims[0],),
                )
                dim2_v[:] = g.y_at_v_locations.values[:]
                dim2_v.setncattr("units", g.y_at_v_locations.attrs["units"])
            except ValueError:
                pass

            # z-axis
            dim3_name = dataset.createVariable("dim3_name", str, ("str_dim",))
            dim3_name[:] = np.array([g.z.dims[0]], dtype="object")
            dim3 = dataset.createVariable(g.z.dims[0], g.z.values.dtype, (g.z.dims[0],))
            dim3[:] = g.z.values[:]
            dim3.setncattr("units", g.z.attrs["units"])
            try:
                dim3_hl = dataset.createVariable(
                    g.z_on_interface_levels.dims[0],
                    g.z_on_interface_levels.values.dtype,
                    (g.z_on_interface_levels.dims[0],),
                )
                dim3_hl[:] = g.z_on_interface_levels.values[:]
                dim3_hl.setncattr("units", g.z_on_interface_levels.attrs["units"])
            except ValueError:
                pass

            # vertical interface level
            z_interface = dataset.createVariable(
                "z_interface", g.z_interface.values.dtype, ("scalar_dim",)
            )
            z_interface[:] = g.z_interface.values.item()
            z_interface.setncattr("units", g.z_interface.attrs["units"])

            # type of horizontal boundary conditions
            hb = self._domain.horizontal_boundary
            hb_type = dataset.createVariable(
                "horizontal_boundary_type", str, ("str_dim",)
            )
            hb_type[:] = np.array(hb.type, dtype="object")

            # the number of boundary layers
            nb = dataset.createVariable("nb", int, ("scalar_dim",))
            nb[:] = np.array([hb.nb], dtype=int)

            # the keyword arguments used to instantiate the object handling the
            # lateral boundary conditions
            keys = []
            for key in hb.kwargs.keys():
                hb_key = "hb_" + key
                value = hb.kwargs[key]

                if isinstance(value, (int, float)):
                    var = dataset.createVariable(hb_key, type(value), ("scalar_dim",))
                    var[:] = np.array([value], dtype=type(value))
                elif isinstance(value, ZhaoSolutionFactory):
                    # TODO: this actually does not work, because only primitive types
                    # TODO: can be stored in a netCDF dataset
                    # var    = dataset.createVariable(hb_key, ZhaoSolutionFactory, ('functor_dim',))
                    # var[:] = np.array([value], dtype='object')
                    pass

                keys.append(hb_key)

            # list of keyword parameter names used to instantiate the object handling
            # the lateral boundary conditions
            dataset.createDimension("strvec2_dim", len(keys))
            hb_kwargs = dataset.createVariable(
                "horizontal_boundary_kwargs", str, ("strvec2_dim",)
            )
            hb_kwargs[:] = np.array(keys, dtype="object")

            # topography type
            topo = g.topography
            topo_type = dataset.createVariable("topography_type", str, ("str_dim",))
            topo_type[:] = np.array([topo.type], dtype="object")

            # keyword arguments used to instantiate the topography
            keys = []
            for key in topo.kwargs.keys():
                topo_key = "topo_" + key
                value = topo.kwargs[key]

                if isinstance(value, sympl.DataArray):
                    var = dataset.createVariable(
                        topo_key, value.values.dtype, ("scalar_dim",)
                    )
                    var[:] = value.values.item()
                    var.setncattr("units", value.attrs["units"])
                elif isinstance(value, str):
                    var = dataset.createVariable(topo_key, str, ("str_dim",))
                    var[:] = np.array([value], dtype="object")
                elif isinstance(value, bool):
                    var = dataset.createVariable(topo_key, int, ("bool_dim",))
                    var[:] = np.array([1 if value else 0], dtype=bool)
                elif isinstance(value, timedelta) or isinstance(value, Timedelta):
                    var = dataset.createVariable(topo_key, float, ("timedelta_dim",))
                    var[:] = np.array([value.total_seconds()], dtype=float)
                    var.setncattr("units", "s")

                keys.append(topo_key)

            # list of keyword parameter names used to instantiate the topography
            dataset.createDimension("strvec3_dim", len(keys))
            topo_kwargs = dataset.createVariable(
                "topography_kwargs", str, ("strvec3_dim",)
            )
            topo_kwargs[:] = np.array(keys, dtype="object")


def load_netcdf_dataset(filename):
    """
    Load the sequence of states stored in a NetCDF dataset,
    and build the underlying domain.

    Parameters
    ----------
    filename : str
        Path to the NetCDF dataset.

    Returns
    -------
    domain : tasmania.Domain
        The underlying domain.
    grid_type : str
        The type of the underlying grid over which the states are defined.
        Either 'physical' or 'numerical'.
    states : list[dict]
        The list of state dictionaries stored in the NetCDF file.
    """
    with xr.open_dataset(filename) as dataset:
        return load_domain(dataset), load_grid_type(dataset), load_states(dataset)


def load_domain(dataset):
    # x-axis
    dims_x = dataset.data_vars["dim1_name"].values.item()
    x = dataset.coords[dims_x]
    domain_x = sympl.DataArray(
        [x.values[0], x.values[-1]], dims=[dims_x], attrs={"units": x.attrs["units"]}
    )
    nx = x.shape[0]

    # y-axis
    dims_y = dataset.data_vars["dim2_name"].values.item()
    y = dataset.coords[dims_y]
    domain_y = sympl.DataArray(
        [y.values[0], y.values[-1]], dims=[dims_y], attrs={"units": y.attrs["units"]}
    )
    ny = y.shape[0]

    # z-axis
    dims_z = dataset.data_vars["dim3_name"].values.item()
    try:
        z_hl = dataset.coords[dims_z + "_on_interface_levels"]
    except KeyError:
        z_hl = sympl.DataArray(np.array((0, 1)), dims=[dims_z], attrs={"units": "1"})
    domain_z = sympl.DataArray(
        [z_hl.values[0], z_hl.values[-1]],
        dims=[dims_z],
        attrs={"units": z_hl.attrs["units"]},
    )
    nz = z_hl.shape[0] - 1

    # vertical interface level
    z_interface = sympl.DataArray(dataset.data_vars["z_interface"])

    # horizontal boundary type
    hb_type = dataset.data_vars["horizontal_boundary_type"].values.item()

    # number of lateral boundary layers
    nb = dataset.data_vars["nb"].values.item()

    # horizontal boundary keyword arguments
    keys = dataset.data_vars["horizontal_boundary_kwargs"].values[:]
    hb_kwargs = {}
    for hb_key in keys:
        try:
            val = dataset.data_vars[hb_key]
            key = hb_key[3:]

            if isinstance(val.values.item(), int):
                hb_kwargs[key] = val.values.item()
        except KeyError:
            pass

    # topography type
    topo_type = dataset.data_vars["topography_type"].values.item()

    # topography keyword arguments
    keys = dataset.data_vars["topography_kwargs"].values[:]
    topo_kwargs = {}
    for topo_key in keys:
        val = dataset.data_vars[topo_key]
        key = topo_key[5:]

        if isinstance(val.values.item(), (str, bool)):
            topo_kwargs[key] = val.values.item()
        elif isinstance(val.values.item(), int):
            topo_kwargs[key] = bool(val.values.item())
        elif val.dims[0] == "timedelta_dim":
            topo_kwargs[key] = Timedelta(seconds=val.values.item())
        else:
            topo_kwargs[key] = sympl.DataArray(val, attrs={"units": val.attrs["units"]})

    return Domain(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        z_interface,
        horizontal_boundary_type=hb_type,
        nb=nb,
        horizontal_boundary_kwargs=hb_kwargs,
        topography_type=topo_type,
        topography_kwargs=topo_kwargs,
        dtype=domain_z.values.dtype,
    )


def load_grid_type(dataset):
    return dataset.data_vars["grid_type"].values.item()


def load_states(dataset):
    names = dataset.data_vars["state_variable_names"].values
    nt = dataset.data_vars[names[0]].shape[0]

    states = []
    for n in range(nt):
        try:
            state = {"time": convert_datetime64_to_datetime(dataset["time"][n])}
        except TypeError:
            state = {
                "time": convert_datetime64_to_datetime(dataset["time"][n].values.item())
            }

        for name in names:
            state[name] = sympl.DataArray(dataset.data_vars[name][n, :, :, :])

        states.append(state)

    return states
