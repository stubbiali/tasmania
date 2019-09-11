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
"""
This module contains:
	DiagnosticComponent
	ImplicitTendencyComponent
	Stepper
	TendencyComponent
"""
from copy import deepcopy
import sympl._core.base_components
from sympl._core.get_np_arrays import get_numpy_arrays_with_properties
from sympl._core.restore_dataarray import restore_data_arrays_with_properties


allowed_grid_types = ("physical", "numerical")


def get_gt_storages(state, property_dict):
    raw_arrays = get_numpy_arrays_with_properties(state, property_dict)

    return_dict = {}
    for key in state:
        if key == "time":
            return_dict["time"] = state["time"]
        else:
            try:
                if property_dict[key]["units"] != state[key].attrs["units"]:
                    return_dict[key] = deepcopy(state[key].attrs["gt_storage"])
                    return_dict[key].data[...] = raw_arrays[key][...]
                else:
                    return_dict[key] = state[key].attrs["gt_storage"]
            except KeyError:
                raise KeyError(
                    "gt_storage key missing in attrs dictionary for {}.".format(key)
                )

    return return_dict


def restore_data_arrays(
    gt_storages,
    output_properties,
    input_state,
    input_properties,
    ignore_names=None,
    ignore_missing=False,
):
    np_arrays = {key: value.data for key, value in gt_storages.items() if key != "time"}
    return_dict = restore_data_arrays_with_properties(
        np_arrays,
        output_properties,
        input_state,
        input_properties,
        ignore_names=ignore_names,
        ignore_missing=ignore_missing,
    )

    for key, value in return_dict.items():
        value.attrs["gt_storage"] = gt_storages[key]

    if "time" in gt_storages:
        return_dict["time"] = gt_storages["time"]

    return return_dict


sympl._core.base_components.__dict__[
    "get_numpy_arrays_with_properties"
] = get_gt_storages
sympl._core.base_components.__dict__[
    "restore_data_arrays_with_properties"
] = restore_data_arrays


class DiagnosticComponent(sympl._core.base_components.DiagnosticComponent):
    """
	Customized version of :class:`sympl.DiagnosticComponent` which keeps track
	of the grid over which the component is instantiated.
	"""

    def __init__(self, domain, grid_type="numerical"):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).
		"""
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid = (
            domain.physical_grid if grid_type == "physical" else domain.numerical_grid
        )
        self._hb = domain.horizontal_boundary
        super().__init__()

    @property
    def grid(self):
        """
		Returns
		-------
		tasmania.Grid :
			The underlying grid.
		"""
        return self._grid

    @property
    def horizontal_boundary(self):
        """
		Returns
		-------
		tasmania.HorizontalBoundary :
			The object handling the lateral boundary conditions.
		"""
        return self._hb


class ImplicitTendencyComponent(sympl._core.base_components.ImplicitTendencyComponent):
    """
	Customized version of :class:`sympl.ImplicitTendencyComponent` which keeps track
	of the grid over which the component is instantiated.
	"""

    def __init__(
        self, domain, grid_type="numerical", tendencies_in_diagnostics=False, name=None
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		tendencies_in_diagnostics : `bool`, optional
			A boolean indicating whether this object will put tendencies of
			quantities in its diagnostic output.
		name : `str`, optional
			A label to be used for this object, for example as would be used for
			Y in the name "X_tendency_from_Y". By default the class name in
			lowercase is used.
		"""
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid = (
            domain.physical_grid if grid_type == "physical" else domain.numerical_grid
        )
        self._hb = domain.horizontal_boundary
        super().__init__(tendencies_in_diagnostics, name)

    @property
    def grid(self):
        """
		Returns
		-------
		tasmania.Grid :
			The underlying grid.
		"""
        return self._grid

    @property
    def horizontal_boundary(self):
        """
		Returns
		-------
		tasmania.HorizontalBoundary :
			The object handling the lateral boundary conditions.
		"""
        return self._hb


class NGTendencyComponent(sympl._core.base_components.TendencyComponent):
    def __init__(
        self, domain, grid_type="numerical", tendencies_in_diagnostics=False, name=None
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).
		"""
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid = (
            domain.physical_grid if grid_type == "physical" else domain.numerical_grid
        )
        self._hb = domain.horizontal_boundary
        super().__init__(tendencies_in_diagnostics=tendencies_in_diagnostics, name=name)

    @property
    def grid(self):
        """
		Returns
		-------
		tasmania.Grid :
			The underlying grid.
		"""
        return self._grid

    @property
    def horizontal_boundary(self):
        """
		Returns
		-------
		tasmania.HorizontalBoundary :
			The object handling the lateral boundary conditions.
		"""
        return self._hb
