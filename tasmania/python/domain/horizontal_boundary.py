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
    """ Handle boundary conditions for a two-dimensional rectilinear grid. """

    register = {}

    def __init__(
        self,
        nx: int,
        ny: int,
        nb: int,
        gt_powered: bool,
        backend: str,
        dtype: taz_types.dtype_t,
    ) -> None:
        """
        Parameters
        ----------
        nx : int
            Number of mass points featured by the physical grid
            along the first dimension.
        ny : int
            Number of mass points featured by the physical grid
            along the second dimension.
        nb : int
            Number of boundary layers.
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : str
            The GT4Py backend.
        dtype : data-type
            The data type of the storages.
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
        Number of mass points featured by the physical grid
        along the first dimension.
        """
        return self._nx

    @property
    def ny(self) -> int:
        """
        Number of mass points featured by the physical grid
        along the second dimension.
        """
        return self._ny

    @property
    def nb(self) -> int:
        """ Number of boundary layers. """
        return self._nb

    @property
    @abc.abstractmethod
    def ni(self) -> int:
        """
        Number of mass points featured by the numerical grid
        along the first dimension.
        """
        pass

    @property
    @abc.abstractmethod
    def nj(self) -> int:
        """
        Number of mass points featured by the numerical grid
        along the second dimension.
        """
        pass

    @property
    def type(self) -> str:
        """
        The string passed to :meth:`tasmania.HorizontalBoundary.factory`
        as ``boundary_type`` argument.
        """
        return self._type

    @type.setter
    def type(self, type_str: str) -> None:
        self._type = type_str

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        The keyword arguments used to initialize the derived class.
        """
        return self._kwargs

    @property
    def reference_state(self) -> taz_types.dataarray_dict_t:
        """
        The reference model state dictionary, defined over the
        numerical grid.
        """
        return self._ref_state if self._ref_state is not None else {}

    @reference_state.setter
    def reference_state(self, ref_state: taz_types.dataarray_dict_t) -> None:
        for name in ref_state:
            if name != "time":
                assert (
                    "units" in ref_state[name].attrs
                ), f"Field {name} of reference state misses units attribute."

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
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the physical grid along the first dimension.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimensions of the returned :class:`~sympl.DataArray`. If not
            specified, the returned :class:`~sympl.DataArray` will have the same
            dimensions of the input one.

        Return
        ------
        sympl.DataArray :
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the numerical grid along the first dimension.
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
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the physical grid along the second dimension.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimensions of the returned :class:`~sympl.DataArray`. If not
            specified, the returned :class:`~sympl.DataArray` will have the same
            dimensions of the input one.

        Return
        ------
        sympl.DataArray :
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the numerical grid along the second dimension.
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
            A raw field defined over the physical grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        numpy.ndarray :
            The same field defined over the numerical grid.
        """
        pass

    @abc.abstractmethod
    def get_physical_xaxis(
        self, caxis: DataArray, dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
        caxis : sympl.DataArray
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the numerical grid along the first dimension.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimensions of the returned :class:`~sympl.DataArray`. If not
            specified, the returned :class:`~sympl.DataArray` will have the same
            dimensions of the input one.

        Return
        ------
        sympl.DataArray :
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the physical grid along the first dimension.
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
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            grid points of the numerical grid along the second dimension.
            Both unstaggered and staggered grid locations are supported.
        dims : `str`, optional
            The dimensions of the returned :class:`~sympl.DataArray`. If not
            specified, the returned :class:`~sympl.DataArray` will have the same
            dimensions of the input one.

        Return
        ------
        sympl.DataArray :
            1-D :class:`sympl.DataArray` collecting the coordinates of the
            grid points of the physical grid along the second dimension.
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
            A raw field defined over the numerical grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        numpy.ndarray :
            The same field defined over the physical grid.
        """
        pass

    @abc.abstractmethod
    def enforce_field(
        self,
        field: taz_types.array_t,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[taz_types.datetime_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """ Enforce the boundary conditions on a raw field.

        The field must be defined on the numerical grid, and gets modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Time point at which the field is defined.
        grid : `tasmania.NumericalGrid`, optional
            The underlying numerical grid.
        """
        pass

    def enforce_raw(
        self,
        state: taz_types.array_dict_t,
        field_properties: Optional[taz_types.properties_mapping_t] = None,
        grid: "Optional[NumericalGrid]" = None,
    ) -> None:
        """ Enforce the boundary conditions on a raw state.

        The state must be defined on the numerical grid, and gets modified in-place.

        Parameters
        ----------
        state : dict[str, array_like]
            Dictionary whose keys are strings denoting model state
            variables, and whose values are :class:`numpy.ndarray`-like duck
            arrays storing the grid values for those variables.
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
        """ Enforce the boundary conditions on a state.

        The state must be defined on the numerical grid, and gets modified in-place.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting model state
            variables, and whose values are :class:`sympl.DataArray`\s
            storing grid values for those variables.
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
        """ Set the outermost layers along the first dimension.

        The field must be x-staggered and defined over the numerical grid.
        It gets modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Time point at which the field is defined.
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
        """ Set the outermost layers along the first dimension.

        The field must be x-staggered and defined over the numerical grid.
        It gets modified in-place.

        Parameters
        ----------
        field : array_like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Time point at which the field is defined.
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
        """ Get an instance of a derived class.

        Parameters
        ----------
        boundary_type : str
            The boundary type, i.e. the string used to register the subclass
            which should be instantiated.
        nx : int
            Number of points featured by the physical grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the physical grid
            along the second horizontal dimension.
        nb : int
            Number of boundary layers.
        gt_powered : `bool`, optional
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : `str`, optional
            The GT4Py backend.
        dtype : `data-type`, optional
            The data type of the storages.
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

        if boundary_type in HorizontalBoundary.register:
            obj = HorizontalBoundary.register[boundary_type](*args, **kwargs)
            obj.type = boundary_type
            return obj
        else:
            raise RuntimeError(
                f"Unknown boundary type '{boundary_type}'. "
                f"Supported types are: "
                f"{', '.join(key for key in HorizontalBoundary.register.keys())}."
            )


def registry(name):
    def core(cls):
        if (
            name in HorizontalBoundary.register
            and HorizontalBoundary.register[name] != cls
        ):
            import warnings

            warnings.warn(
                f"Cannot register '{name}' as already present in HorizontalBoundary.register."
            )
        else:
            HorizontalBoundary.register[name] = cls
        return cls

    return core
