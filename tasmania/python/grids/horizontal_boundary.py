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
from copy import deepcopy
import numpy as np
from sympl import DataArray
from typing import Any, Dict, Optional, Sequence

from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import deepcopy_dataarray


class HorizontalBoundary(abc.ABC):
    """
    Abstract base class whose children handle the
    horizontal boundary conditions.
    """

    def __init__(
        self, nx: int, ny: int, nb: int, gt_powered: bool, backend: str, dtype: taz_types.dtype_t
    ) -> None:
        """
        Parameters
        ----------
        nx : int
            Number of mass points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of mass points featured by the *physical* grid
            along the second horizontal dimension.
        nb : int
            Number of boundary layers.
        gt_powered : bool
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : str
            The GT4Py backend.
        dtype : data-type
            Data type of the storages.
        """
        self._nx = nx
        self._ny = ny
        self._nb = nb

        self._gt_powered = gt_powered
        self._backend = backend
        self._dtype = dtype

        self._type = ""
        self._kwargs = {}
        self._ref_state = None

    @property
    def nx(self) -> int:
        """
        Return
        ------
        int :
            Number of mass points featured by the *physical* grid
            along the first horizontal dimension.
        """
        return self._nx

    @property
    def ny(self) -> int:
        """
        Return
        ------
        int :
            Number of mass points featured by the *physical* grid
            along the second horizontal dimension.
        """
        return self._ny

    @property
    def nb(self) -> int:
        """
        Return
        ------
        int :
            Number of boundary layers.
        """
        return self._nb

    @property
    @abc.abstractmethod
    def ni(self) -> int:
        """
        Return
        ------
        int :
            Number of mass points featured by the *numerical* grid
            along the first horizontal dimension.
        """
        pass

    @property
    @abc.abstractmethod
    def nj(self) -> int:
        """
        Return
        ------
        int :
            Number of mass points featured by the *numerical* grid
            along the second horizontal dimension.
        """
        pass

    @property
    def type(self) -> str:
        """
        Return
        ------
        str :
            The string passed to :meth:`tasmania.HorizontalBoundary.factory`
            as `boundary_type` argument.
        """
        return self._type

    @type.setter
    def type(self, type_str: str) -> None:
        """
        Parameters
        ----------
        type_str : str
            The string passed to :meth:`tasmania.HorizontalBoundary.factory`
            as `boundary_type` argument.
        """
        self._type = type_str

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Return
        ------
        dict :
            The keyword arguments passed to the constructor of the derived class.
        """
        return self._kwargs

    @property
    def reference_state(self) -> taz_types.dataarray_dict_t:
        """
        Return
        ------
        dict[str, sympl.DataArray] :
            The reference model state dictionary, defined over the
            numerical grid.
        """
        return self._ref_state if self._ref_state is not None else {}

    @reference_state.setter
    def reference_state(self, ref_state: taz_types.dataarray_dict_t) -> None:
        """
        Parameters
        ----------
        ref_state : dict[str, sympl.DataArray]
            The reference model state dictionary.
        """
        for name in ref_state:
            if name != "time":
                assert (
                    "units" in ref_state[name].attrs
                ), "Field {} of reference state misses units attribute.".format(name)

        self._ref_state = {}
        for name in ref_state:
            if name == "time":
                self._ref_state["time"] = deepcopy(ref_state["time"])
            else:
                self._ref_state[name] = deepcopy_dataarray(ref_state[name])

    @abc.abstractmethod
    def get_numerical_xaxis(
        self, paxis: DataArray, dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
        paxis : sympl.DataArray
            1-D :class:`sympl.DataArray` representing the axis along the
            first horizontal dimension of the physical domain.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimension of the returned axis. If not specified, the
            returned axis will have the same dimension of the input axis.

        Return
        ------
        sympl.DataArray :
            1-D :class:`sympl.DataArray` representing the associated
            numerical axis.
        """
        pass

    @abc.abstractmethod
    def get_numerical_yaxis(
        self, paxis: DataArray, dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
        paxis : sympl.DataArray
            1-D :class:`sympl.DataArray` representing the axis along the
            second horizontal dimension of the physical domain.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimension of the returned axis. If not specified, the
            returned axis will have the same dimension of the input axis.

        Return
        ------
        sympl.DataArray :
            1-D :class:`sympl.DataArray` representing the associated
            numerical axis.
        """
        pass

    @abc.abstractmethod
    def get_numerical_field(
        self, field: taz_types.array_t, field_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        field : array_like
            A field defined over the *physical* grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        numpy.ndarray :
            The same field defined over the *numerical* grid.
        """

    @abc.abstractmethod
    def get_physical_xaxis(
        self, caxis: DataArray, dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
        caxis : sympl.DataArray
            1-D :class:`sympl.DataArray` representing the axis along the
            first horizontal dimension of the numerical domain.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimension of the returned axis. If not specified, the
            returned axis will have the same dimension of the input axis.

        Return
        ------
        sympl.DataArray :
            1-D :class:`sympl.DataArray` representing the associated
            physical axis.
        """
        pass

    @abc.abstractmethod
    def get_physical_yaxis(
        self, caxis: DataArray, dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
        caxis : sympl.DataArray
            1-D :class:`sympl.DataArray` representing the axis along the
            second horizontal dimension of the numerical domain.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimension of the returned axis. If not specified, the
            returned axis will have the same dimension of the input axis.

        Return
        ------
        sympl.DataArray :
            1-D :class:`sympl.DataArray` representing the associated
            physical axis.
        """
        pass

    @abc.abstractmethod
    def get_physical_field(
        self, field: taz_types.array_t, field_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        field : array_like
            A raw field defined over the *numerical* grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        numpy.ndarray :
            The same field defined over the *physical* grid.
        """

    @abc.abstractmethod
    def enforce_field(
        self,
        field: taz_types.array_t,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[taz_types.datetime_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """
        Enforce the horizontal boundary conditions on the passed field,
        which is modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Temporal instant at which the field is defined.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """

    def enforce_raw(
        self,
        state: taz_types.array_dict_t,
        field_properties: Optional[taz_types.properties_mapping_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """
        Enforce the horizontal boundary conditions on the passed state,
        which is modified in-place.

        Parameters
        ----------
        state : dict[str, array_like]
            Dictionary whose keys are strings denoting model state
            variables, and whose values are :class:`numpy.ndarray`-like arrays
            storing values for those variables.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings denoting the model variables
            on which boundary conditions should be enforced, and whose
            values are dictionaries specifying fundamental properties (units)
            of those fields. If not specified, boundary conditions are
            enforced on all model variables included in the model state.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """
        rfps = {
            name: {"units": self.reference_state[name].attrs["units"]}
            for name in self.reference_state
            if name != "time"
        }
        fps = (
            rfps
            if field_properties is None
            else {key: val for key, val in field_properties.items() if key in rfps}
        )

        fns = tuple(name for name in state if name != "time" and name in fps)

        time = state.get("time", None)

        for field_name in fns:
            field_units = fps[field_name].get("units", rfps[field_name]["units"])
            self.enforce_field(
                state[field_name],
                field_name=field_name,
                field_units=field_units,
                time=time,
                grid=grid,
            )

    def enforce(
        self,
        state: taz_types.dataarray_dict_t,
        field_names: Optional[Sequence[str]] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """
        Enforce the horizontal boundary conditions on the passed state,
        which is modified in-place.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting model state
            variables, and whose values are :class:`sympl.DataArray`\s
            storing values for those variables.
        field_names : `tuple[str]`, optional
            Tuple of strings denoting the model variables on which
            boundary conditions should be enforced. If not specified,
            boundary conditions are enforced on all model variables
            included in the model state.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """
        fns = (
            tuple(name for name in self.reference_state if name != "time")
            if field_names is None
            else tuple(name for name in field_names if name in self.reference_state)
        )

        fns = tuple(name for name in state if name in fns)

        time = state.get("time", None)

        for field_name in fns:
            try:
                field_units = state[field_name].attrs["units"]
            except KeyError:
                raise KeyError("Field {} misses units attribute.".format(field_name))
            self.enforce_field(
                state[field_name].values,
                field_name=field_name,
                field_units=field_units,
                time=time,
                grid=grid,
            )

    @abc.abstractmethod
    def set_outermost_layers_x(
        self,
        field: taz_types.array_t,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[taz_types.datetime_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """
        Set the outermost layers along the first horizontal dimension of a
        x-staggered field, which is modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Temporal instant at which the field is defined.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """
        pass

    @abc.abstractmethod
    def set_outermost_layers_y(
        self,
        field: taz_types.array_t,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[taz_types.datetime_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """
        Set the outermost layers along the second horizontal dimension of a
        x-staggered field, which is modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Temporal instant at which the field is defined.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """
        pass

    @staticmethod
    def factory(
        boundary_type: str,
        nx: int,
        ny: int,
        nb: int,
            gt_powered: bool = True,
            *,
        backend: str = "numpy",
        dtype: taz_types.dtype_t = np.float64,
        **kwargs
    ) -> "HorizontalBoundary":
        """
        Parameters
        ----------
        boundary_type : str
            The boundary type, identifying the child class to instantiate.
            Available options are:

                * 'relaxed';
                * 'periodic';
                * 'dirichlet';
                * 'identity'.

        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension.
        nb : int
            Number of boundary layers.
        gt_powered : `bool`, optional
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        dtype : `data-type`, optional
            Data type of the storages.
        kwargs :
            Keyword arguments to be directly forwarded to the
            constructor of the child class.

        Returns
        -------
        obj :
            An object of the suitable child class.
        """
        args = (nx, ny, nb, gt_powered)
        child_kwargs = {"backend": backend, "dtype": dtype}
        child_kwargs.update(kwargs)

        import tasmania.python.grids._horizontal_boundary as module

        if boundary_type == "relaxed":
            if ny == 1:
                obj = module.Relaxed1DX(*args, **child_kwargs)
            elif nx == 1:
                obj = module.Relaxed1DY(*args, **child_kwargs)
            else:
                obj = module.Relaxed(*args, **child_kwargs)
        elif boundary_type == "periodic":
            if ny == 1:
                obj = module.Periodic1DX(*args, **child_kwargs)
            elif nx == 1:
                obj = module.Periodic1DY(*args, **child_kwargs)
            else:
                obj = module.Periodic(*args, **child_kwargs)
        elif boundary_type == "dirichlet":
            if ny == 1:
                obj = module.Dirichlet1DX(*args, **child_kwargs)
            elif nx == 1:
                obj = module.Dirichlet1DY(*args, **child_kwargs)
            else:
                obj = module.Dirichlet(*args, **child_kwargs)
        elif boundary_type == "identity":
            if ny == 1:
                obj = module.Identity1DX(*args, **child_kwargs)
            elif nx == 1:
                obj = module.Identity1DY(*args, **child_kwargs)
            else:
                obj = module.Identity(*args, **child_kwargs)
        else:
            raise ValueError(
                "Unknown boundary type {}. Supported types are {}.".format(
                    boundary_type,
                    ",".join(("relaxed", "periodic", "dirichlet", "identity")),
                )
            )

        obj.type = boundary_type

        return obj
