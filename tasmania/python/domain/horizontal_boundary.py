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
from sympl import DataArray
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from tasmania.python.domain.grid import Grid, NumericalGrid
from tasmania.python.framework.register import factorize
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.storage import deepcopy_dataarray

if TYPE_CHECKING:
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class HorizontalBoundary(StencilFactory, abc.ABC):
    """Handle boundary conditions for a two-dimensional rectilinear grid."""

    registry = {}

    def __init__(
        self: "HorizontalBoundary",
        grid: Grid,
        nb: int,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional grid.
        nb : int
            Number of boundary layers.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(backend, backend_options, storage_options)

        self._pgrid = grid
        self._nb = nb
        self._storage_shape = storage_shape

        self._ngrid = NumericalGrid(self)

        self._type = ""
        self._kwargs = {}
        self._ref_state = None

    @property
    def kwargs(self: "HorizontalBoundary") -> Dict[str, Any]:
        """The keyword arguments used to initialize the derived class."""
        return self._kwargs

    @property
    def nb(self: "HorizontalBoundary") -> int:
        """Number of boundary layers."""
        return self._nb

    @property
    def numerical_grid(self: "HorizontalBoundary") -> NumericalGrid:
        """The underlying numerical grid."""
        return self._ngrid

    @property
    def nx(self: "HorizontalBoundary") -> int:
        """
        Number of mass points featured by the physical grid
        along the first dimension.
        """
        return self._pgrid.nx

    @property
    def ny(self: "HorizontalBoundary") -> int:
        """
        Number of mass points featured by the physical grid
        along the second dimension.
        """
        return self._pgrid.ny

    @property
    def physical_grid(self: "HorizontalBoundary") -> Grid:
        """The underlying physical grid."""
        return self._pgrid

    @property
    def reference_state(self: "HorizontalBoundary") -> ty.DataArrayDict:
        """
        The reference model state dictionary, defined over the
        numerical grid.
        """
        return self._ref_state if self._ref_state is not None else {}

    @reference_state.setter
    def reference_state(
        self: "HorizontalBoundary", ref_state: ty.DataArrayDict
    ) -> None:
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

    @property
    def type(self: "HorizontalBoundary") -> str:
        """
        The string passed to :meth:`tasmania.HorizontalBoundary.factory`
        as ``boundary_type`` argument.
        """
        return self._type

    @type.setter
    def type(self: "HorizontalBoundary", value: str) -> None:
        self._type = value

    @property
    @abc.abstractmethod
    def ni(self: "HorizontalBoundary") -> int:
        """
        Number of mass points featured by the numerical grid
        along the first dimension.
        """
        pass

    @property
    @abc.abstractmethod
    def nj(self: "HorizontalBoundary") -> int:
        """
        Number of mass points featured by the numerical grid
        along the second dimension.
        """
        pass

    @abc.abstractmethod
    def get_numerical_xaxis(
        self: "HorizontalBoundary", dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
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
    def get_numerical_xaxis_staggered(
        self: "HorizontalBoundary", dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
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
        self: "HorizontalBoundary", dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
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
    def get_numerical_yaxis_staggered(
        self: "HorizontalBoundary", dims: Optional[str] = None
    ) -> DataArray:
        """
        Parameters
        ----------
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
        self: "HorizontalBoundary",
        field: ty.Storage,
        field_name: Optional[str] = None,
    ) -> ty.Storage:
        """
        Parameters
        ----------
        field : array-like
            A raw field defined over the physical grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        array-like :
            The same field defined over the numerical grid.
        """
        pass

    @abc.abstractmethod
    def get_physical_field(
        self: "HorizontalBoundary",
        field: ty.Storage,
        field_name: Optional[str] = None,
    ) -> ty.Storage:
        """
        Parameters
        ----------
        field : array-like
            A raw field defined over the numerical grid.
        field_name : `str`, optional
            The field name.

        Return
        ------
        array-like :
            The same field defined over the physical grid.
        """
        pass

    @abc.abstractmethod
    def enforce_field(
        self: "HorizontalBoundary",
        field: ty.Storage,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[ty.Datetime] = None,
    ) -> None:
        """Enforce the boundary conditions on a raw field.

        The field must be defined on the numerical grid, and gets modified
        in-place.

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
        """
        pass

    def enforce_raw(
        self: "HorizontalBoundary",
        state: ty.StorageDict,
        field_properties: Optional[ty.properties_mapping_t] = None,
    ) -> None:
        """Enforce the boundary conditions on a raw state.

        The state must be defined on the numerical grid, and gets modified
        in-place.

        Parameters
        ----------
        state : dict[str, array-like]
            Dictionary whose keys are strings denoting model state
            variables, and whose values are :class:`numpy.ndarray`-like duck
            arrays storing the grid values for those variables.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings denoting the model variables
            on which boundary conditions should be enforced, and whose
            values are dictionaries specifying fundamental properties (units)
            of those fields. If not specified, boundary conditions are
            enforced on all model variables included in the model state.
        """
        rfps = {
            name: {"units": self.reference_state[name].attrs["units"]}
            for name in self.reference_state
            if name != "time"
        }
        fps = (
            rfps
            if field_properties is None
            else {
                key: val
                for key, val in field_properties.items()
                if key in rfps
            }
        )

        fns = tuple(name for name in state if name != "time" and name in fps)

        time = state.get("time", None)

        for field_name in fns:
            field_units = fps[field_name].get(
                "units", rfps[field_name]["units"]
            )
            self.enforce_field(
                state[field_name],
                field_name=field_name,
                field_units=field_units,
                time=time,
            )

    def enforce(
        self: "HorizontalBoundary",
        state: ty.DataArrayDict,
        field_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Enforce the boundary conditions on a state.

        The state must be defined on the numerical grid, and gets modified
        in-place.

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
        """
        fns = (
            tuple(name for name in self.reference_state if name != "time")
            if field_names is None
            else tuple(
                name for name in field_names if name in self.reference_state
            )
        )

        fns = tuple(name for name in state if name in fns)

        time = state.get("time", None)

        for field_name in fns:
            try:
                field_units = state[field_name].attrs["units"]
            except KeyError:
                raise KeyError(f"Field {field_name} misses units attribute.")
            self.enforce_field(
                state[field_name].data,
                field_name=field_name,
                field_units=field_units,
                time=time,
            )

    @abc.abstractmethod
    def set_outermost_layers_x(
        self: "HorizontalBoundary",
        field: ty.Storage,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[ty.Datetime] = None,
    ) -> None:
        """Set the outermost layers along the first dimension.

        The field must be x-staggered and defined over the numerical grid.
        It gets modified in-place.

        Parameters
        ----------
        field : array-like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Time point at which the field is defined.
        """
        pass

    @abc.abstractmethod
    def set_outermost_layers_y(
        self,
        field: ty.Storage,
        field_name: Optional[str] = None,
        field_units: Optional[str] = None,
        time: Optional[ty.Datetime] = None,
    ) -> None:
        """Set the outermost layers along the first dimension.

        The field must be x-staggered and defined over the numerical grid.
        It gets modified in-place.

        Parameters
        ----------
        field : array-like
            The raw field.
        field_name : `str`, optional
            The field name.
        field_units : `str`, optional
            The field units.
        time : `datetime`, optional
            Time point at which the field is defined.
        """
        pass

    @staticmethod
    def factory(
        boundary_type: str,
        grid: Grid,
        nb: int,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
        **kwargs
    ) -> "HorizontalBoundary":
        """Get an instance of a derived class.

        Parameters
        ----------
        boundary_type : str
            The boundary type, i.e. the string used to register the subclass
            which should be instantiated. Available options are:

            * "dirichlet";
            * "identity";
            * "periodic";
            * "relaxed".

        grid : Grid
            The underlying physical grid.
        nb : int
            Number of boundary layers.
        backend : `str`, optional
            The backend.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `tuple[int]`, optional
            The shape of the storages allocated within this class.
        storage_options : `StorageOptions`, optional
            Storage-related options.

        Returns
        -------
        obj :
            An object of the suitable child class.
        """
        args = (grid, nb)
        child_kwargs = {
            "backend": backend,
            "backend_options": backend_options,
            "storage_shape": storage_shape,
            "storage_options": storage_options,
        }
        child_kwargs.update(kwargs)
        obj = factorize(boundary_type, HorizontalBoundary, args, child_kwargs)
        obj.type = boundary_type
        return obj
