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
import abc
import numpy as np
from sympl._core.base_components import InputChecker, DiagnosticChecker
from sympl._core.get_np_arrays import get_numpy_arrays_with_properties
from sympl._core.restore_dataarray import restore_data_arrays_with_properties
from typing import Mapping, Optional, Sequence, Tuple, Union

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import taz_types
from tasmania.python.utils.utils import assert_sequence


SequenceType = (tuple, list)


class FakeComponent:
    def __init__(
        self, properties: Mapping[str, taz_types.properties_dict_t]
    ) -> None:
        for name, value in properties.items():
            if name == "input_properties":
                self.input_properties = value
            elif name == "tendency_properties":
                self.tendency_properties = value
            elif name == "diagnostic_properties":
                self.diagnostic_properties = value
            elif name == "output_properties":
                self.output_properties = value


class OfflineDiagnosticComponent(abc.ABC):
    """
    Abstract base class whose derived classes retrieve diagnostics
    from multiple state dictionaries.
    """

    def __init__(self) -> None:
        assert isinstance(self.input_properties, SequenceType), (
            "input_properties attribute is of type {}, "
            "but should be a sequence of dict.".format(
                self.input_properties.__class__.__name__
            )
        )

        assert_sequence(self.input_properties, reftype=dict)

        assert isinstance(self.diagnostic_properties, dict), (
            "diagnostic_properties attribute is of type {}, "
            "but should be a dict.".format(
                self.diagnostic_properties.__class__.__name__
            )
        )

        self._input_checkers = []
        for input_property in self.input_properties:
            self._input_checkers.append(
                InputChecker(
                    FakeComponent({"input_properties": input_property})
                )
            )

        self._diagnostic_checker = DiagnosticChecker(
            FakeComponent(
                {
                    "input_properties": self.input_properties[0],
                    "diagnostic_properties": self.diagnostic_properties,
                }
            )
        )

    @property
    @abc.abstractmethod
    def input_properties(self) -> Sequence[taz_types.properties_dict_t]:
        """
        Returns
        -------
        Sequence[dict[str, dict]] :
            Sequence of dictionaries, whose keys are strings denoting
            variables which should be present in the corresponding input
            state, and whose values are dictionaries specifying fundamental
            properties (dims, units) for those variables.
        """
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        """
        Returns
        -------
        dict[str, dict] :
            Dictionary whose keys are strings denoting the output diagnostics,
            and whose values are dictionaries specifying fundamental
            properties (dims, units) for those diagnostics.
        """
        pass

    def __call__(
        self, *states: taz_types.dataarray_dict_t
    ) -> taz_types.dataarray_dict_t:
        """
        Call operator retrieving the diagnostics.

        Parameters
        ----------
        states : Sequence[dict[str, sympl.DataArray]]
            State dictionaries whose keys are strings denoting
            model variables, and whose values are :class:`sympl.DataArray`\s
            representing values for those variables.

        Returns
        -------
        dict[str, sympl.DataArray] :
            Dictionary whose keys are strings denoting diagnostics,
            and whose values are :class:`sympl.DataArray`\s representing
            values for those diagnostics.
        """
        assert_sequence(
            states, reflen=len(self.input_properties), reftype=dict
        )

        for input_checker, state in zip(self._input_checkers, states):
            input_checker.check_inputs(state)

        raw_states = []
        for input_properties, state in zip(self.input_properties, states):
            raw_states.append(
                get_numpy_arrays_with_properties(state, input_properties)
            )
            raw_states[-1]["time"] = state["time"]

        raw_diagnostics = self.array_call(*raw_states)

        # To be safe
        raw_diagnostics_without_time = {
            key: val for key, val in raw_diagnostics.items() if key != "time"
        }

        self._diagnostic_checker.check_diagnostics(
            raw_diagnostics_without_time
        )

        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics_without_time,
            self.diagnostic_properties,
            states[0],
            self.input_properties[0],
        )

        return diagnostics

    @abc.abstractmethod
    def array_call(
        self, *states: taz_types.array_dict_t
    ) -> taz_types.array_dict_t:
        """
        Retrieve the diagnostics.

        Parameters
        ----------
        states : Sequence[dict[str, sympl.DataArray]]
            State dictionaries whose keys are strings denoting
            model variables, and whose values are :class:`numpy.ndarray`-like arrays
            representing raw values for those variables.

        Returns
        -------
        dict[str, array_like] :
            Dictionary whose keys are strings denoting diagnostics,
            and whose values are :class:`numpy.ndarray`-like arrays representing
            raw values for those diagnostics.
        """
        pass


class RMSD(OfflineDiagnosticComponent):
    """
    Offline diagnostic calculating the root-mean-square deviation (RMSD)
    between model variables of two state dictionaries.
    """

    def __init__(
        self,
        grid: Union[Grid, Sequence[Grid]],
        fields: Mapping[str, taz_types.properties_dict_t],
        x: Optional[slice] = None,
        y: Optional[slice] = None,
        z: Optional[slice] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid, Sequence[tasmania.Grid]
            The underlying grid(s). The two input states may be defined
            over different domain.
        fields : dict[str, dict]
            Dictionary whose keys are strings denoting the model variables
            for which the RMSD should be computed, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            for those variables
        x : `slice`, `Sequence[slice]`, optional
            The projection along the first coordinate axis of the slice
            of grid points to be considered for the calculation of the RMSD.
            If not specified, all grid points along the first dimension
            are considered. Different slices for different states are allowed.
        y : `slice`, `Sequence[slice]`, optional
            The projection along the second coordinate axis of the slice
            of grid points to be considered for the calculation of the RMSD.
            If not specified, all grid points along the second dimension
            are considered. Different slices for different states are allowed.
        z : `slice`, `Sequence[slice]`, optional
            The projection along the third coordinate axis of the slice
            of grid points to be considered for the calculation of the RMSD.
            If not specified, all grid points along the third dimension
            are considered. Different slices for different states are allowed.
        """
        self._grids = grid if isinstance(grid, SequenceType) else (grid, grid)
        assert_sequence(self._grids, reflen=2, reftype=Grid)

        self._fields = fields

        super().__init__()

        x = x if isinstance(x, SequenceType) else (x, x)
        y = y if isinstance(y, SequenceType) else (y, y)
        z = z if isinstance(z, SequenceType) else (z, z)

        self._xs = tuple(
            slice(0, None, None) if arg is None else arg for arg in x
        )
        self._ys = tuple(
            slice(0, None, None) if arg is None else arg for arg in y
        )
        self._zs = tuple(
            slice(0, None, None) if arg is None else arg for arg in z
        )

        assert_sequence(self._xs, reflen=2, reftype=slice)
        assert_sequence(self._ys, reflen=2, reftype=slice)
        assert_sequence(self._zs, reflen=2, reftype=slice)

    @property
    def input_properties(
        self,
    ) -> Tuple[taz_types.properties_dict_t, taz_types.properties_dict_t]:
        g1, g2 = self._grids

        return_list = ({}, {})

        for name, units in self._fields.items():
            dimx = (
                g1.x_at_u_locations.dims[0]
                if "u_locations" in name
                else g1.x.dims[0]
            )
            dimy = (
                g1.y_at_v_locations.dims[0]
                if "v_locations" in name
                else g1.y.dims[0]
            )
            dimz = (
                g1.z_on_interface_levels.dims[0]
                if "interface_levels" in name
                else g1.z.dims[0]
            )
            return_list[0][name] = {"dims": (dimx, dimy, dimz), "units": units}

            dimx = (
                g2.x_at_u_locations.dims[0]
                if "u_locations" in name
                else g2.x.dims[0]
            )
            dimy = (
                g2.y_at_v_locations.dims[0]
                if "v_locations" in name
                else g2.y.dims[0]
            )
            dimz = (
                g2.z_on_interface_levels.dims[0]
                if "interface_levels" in name
                else g2.z.dims[0]
            )
            return_list[1][name] = {"dims": (dimx, dimy, dimz), "units": units}

        return return_list

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {
            "rmsd_of_" + name: {"dims": ("scalar",) * 3, "units": units}
            for name, units in self._fields.items()
        }

    def array_call(
        self, state1: taz_types.array_dict_t, state2: taz_types.array_dict_t
    ) -> taz_types.array_dict_t:
        x1, y1, z1 = self._xs[0], self._ys[0], self._zs[0]
        x2, y2, z2 = self._xs[1], self._ys[1], self._zs[1]

        diags = {}

        for name in self._fields.keys():
            arr1 = state1[name][x1, y1, z1]
            arr2 = state2[name][x2, y2, z2]
            tmp = np.linalg.norm(arr1 - arr2) / np.sqrt(arr1.size)
            diags["rmsd_of_" + name] = np.array(tmp)[
                np.newaxis, np.newaxis, np.newaxis
            ]

        return diags


class RRMSD(OfflineDiagnosticComponent):
    """
    Offline diagnostic calculating the relative root-mean-square
    deviation (RRMSD) between model variables of two state dictionaries.
    """

    def __init__(
        self,
        grid: Union[Grid, Sequence[Grid]],
        fields: Mapping[str, taz_types.properties_dict_t],
        x: Optional[slice] = None,
        y: Optional[slice] = None,
        z: Optional[slice] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid, Sequence[tasmania.Grid]
            The underlying grid(s). The two input states may be defined
            over different domain.
        fields : dict[str, dict]
            Dictionary whose keys are strings denoting the model variables
            for which the RRMSD should be computed, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            for those variables
        x : `slice`, `Sequence[slice]`, optional
            The projection along the first coordinate axis of the slice
            of grid points to be considered for the calculation of the RRMSD.
            If not specified, all grid points along the first dimension
            are considered. Different slices for different states are allowed.
        y : `slice`, `Sequence[slice]`, optional
            The projection along the second coordinate axis of the slice
            of grid points to be considered for the calculation of the RRMSD.
            If not specified, all grid points along the second dimension
            are considered. Different slices for different states are allowed.
        z : `slice`, `Sequence[slice]`, optional
            The projection along the third coordinate axis of the slice
            of grid points to be considered for the calculation of the RRMSD.
            If not specified, all grid points along the third dimension
            are considered. Different slices for different states are allowed.
        """
        self._grids = grid if isinstance(grid, SequenceType) else (grid, grid)
        assert_sequence(self._grids, reflen=2, reftype=Grid)

        self._fields = fields

        super().__init__()

        x = x if isinstance(x, SequenceType) else (x, x)
        y = y if isinstance(y, SequenceType) else (y, y)
        z = z if isinstance(z, SequenceType) else (z, z)

        self._xs = tuple(
            slice(0, None, None) if arg is None else arg for arg in x
        )
        self._ys = tuple(
            slice(0, None, None) if arg is None else arg for arg in y
        )
        self._zs = tuple(
            slice(0, None, None) if arg is None else arg for arg in z
        )

        assert_sequence(self._xs, reflen=2, reftype=slice)
        assert_sequence(self._ys, reflen=2, reftype=slice)
        assert_sequence(self._zs, reflen=2, reftype=slice)

    @property
    def input_properties(
        self,
    ) -> Tuple[taz_types.properties_dict_t, taz_types.properties_dict_t]:
        g1, g2 = self._grids

        return_list = ({}, {})

        for name, units in self._fields.items():
            dimx = (
                g1.x_at_u_locations.dims[0]
                if "u_locations" in name
                else g1.x.dims[0]
            )
            dimy = (
                g1.y_at_v_locations.dims[0]
                if "v_locations" in name
                else g1.y.dims[0]
            )
            dimz = (
                g1.z_on_interface_levels.dims[0]
                if "interface_levels" in name
                else g1.z.dims[0]
            )
            return_list[0][name] = {"dims": (dimx, dimy, dimz), "units": units}

            dimx = (
                g2.x_at_u_locations.dims[0]
                if "u_locations" in name
                else g2.x.dims[0]
            )
            dimy = (
                g2.y_at_v_locations.dims[0]
                if "v_locations" in name
                else g2.y.dims[0]
            )
            dimz = (
                g2.z_on_interface_levels.dims[0]
                if "interface_levels" in name
                else g2.z.dims[0]
            )
            return_list[1][name] = {"dims": (dimx, dimy, dimz), "units": units}

        return return_list

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {
            "rrmsd_of_" + name: {"dims": ("scalar",) * 3, "units": "1"}
            for name in self._fields.keys()
        }

    def array_call(
        self, state1: taz_types.array_dict_t, state2: taz_types.array_dict_t
    ) -> taz_types.array_dict_t:
        x1, y1, z1 = self._xs[0], self._ys[0], self._zs[0]
        x2, y2, z2 = self._xs[1], self._ys[1], self._zs[1]

        diags = {}

        for name in self._fields.keys():
            arr1 = state1[name][x1, y1, z1]
            arr2 = state2[name][x2, y2, z2]
            tmp = np.linalg.norm(arr1 - arr2) / np.linalg.norm(arr2)
            diags["rrmsd_of_" + name] = np.array(tmp)[
                np.newaxis, np.newaxis, np.newaxis
            ]

        return diags


class ColumnSum(OfflineDiagnosticComponent):
    """
    Sum the values of a field over each column.
    """

    def __init__(self, grid: Grid, field_name: str, field_units: str):
        self._grid = grid
        self._fname = field_name
        self._funits = field_units
        super().__init__()

    @property
    def input_properties(self) -> Tuple[taz_types.properties_dict_t]:
        g = self._grid
        dimx = (
            g.x_at_u_locations.dims[0]
            if "u_locations" in self._fname
            else g.x.dims[0]
        )
        dimy = (
            g.y_at_v_locations.dims[0]
            if "v_locations" in self._fname
            else g.y.dims[0]
        )
        dimz = (
            g.z_on_interface_levels[0]
            if "interface_levels" in self._fname
            else g.z.dims[0]
        )

        return_list = ({"dims": (dimx, dimy, dimz), "units": self._funits},)

        return return_list

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        g = self._grid
        dimx = (
            g.x_at_u_locations.dims[0]
            if "u_locations" in self._fname
            else g.x.dims[0]
        )
        dimy = (
            g.y_at_v_locations.dims[0]
            if "v_locations" in self._fname
            else g.y.dims[0]
        )
        dimz = (
            g.z_on_interface_levels[0]
            if "interface_levels" in self._fname
            else g.z.dims[0]
        )
        dimz += "_at_surface_level"
        return {"dims": (dimx, dimy, dimz), "units": self._funits}

    def array_call(
        self, state: taz_types.array_dict_t
    ) -> taz_types.array_dict_t:
        field = state[self._fname]
        out = np.zeros((field.shape[0], field.shape[1], 1), dtype=field.dtype)
        np.sum(field, axis=2, out=out)
        return out
