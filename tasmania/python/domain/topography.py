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
from pandas import Timedelta
from sympl import DataArray
from typing import Any, Dict, Optional, TYPE_CHECKING

from tasmania.python.utils import taz_types
from tasmania.python.utils.framework_utils import factorize
from tasmania.python.utils.storage_utils import get_dataarray_2d
from tasmania.python.utils.utils import smaller_than as lt

if TYPE_CHECKING:
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
    from tasmania.python.domain.horizontal_grid import (
        PhysicalHorizontalGrid,
        NumericalHorizontalGrid,
    )


class Topography:
    """ A time-dependent topography.

    Although clearly not physical, a terrain surface (slowly) growing in the
    early stages of a simulation may help to retain numerical stability, as it
    prevents steep gradients in the first few iterations.
    """

    def __init__(
        self,
        steady_profile: DataArray,
        profile: Optional[DataArray] = None,
        time: Optional[taz_types.timedelta_t] = None,
    ) -> None:
        """
        Parameters
        ----------
        steady_profile : sympl.DataArray
            2-D :class:`~sympl.DataArray` representing the steady-state height profile.
        profile : `sympl.DataArray`, optional
            2-D :class:`~sympl.DataArray` representing the current height profile.
        time : `datetime.timedelta`
            The elapsed simulation time after which the topography should stop increasing.
            If not specified, a time-invariant terrain surface-height is assumed.
        """
        self._steady_profile = steady_profile.to_units("m")

        self._time = time or Timedelta(seconds=0)
        self._fact = float(self._time.total_seconds() == 0.0)

        self._profile = profile if profile is not None else deepcopy(steady_profile)
        self._profile.attrs["units"] = "m"
        self._profile.values[...] = self._fact * self._steady_profile.values[...]

    @property
    def profile(self) -> DataArray:
        """
        2-D :class:`~sympl.DataArray` storing the current topography profile.
        """
        return self._profile

    @property
    def steady_profile(self) -> DataArray:
        """
        2-D :class:`~sympl.DataArray` storing the steady-state topography profile.
        """
        return self._steady_profile

    @property
    def time(self) -> taz_types.datetime_t:
        """
        The elapsed simulation time after which the topography stops increasing.
        """
        return self._time

    def update(self, time: taz_types.datetime_t) -> None:
        """
        Update the topography at current simulation time.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        if lt(self._fact, 1.0):
            self._fact = min(time / self.time, 1.0)
            self._profile.values[...] = self._fact * self._steady_profile.values


class PhysicalTopography(abc.ABC, Topography):
    """ A time-dependent topography defined over a physical grid. """

    registry = {}

    def __init__(
        self,
        grid: "PhysicalHorizontalGrid",
        time: taz_types.timedelta_t,
        smooth: bool,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.PhysicalHorizontalGrid
            The underlying :class:`tasmania.PhysicalHorizontalGrid`.
        time : datetime.timedelta
            The elapsed simulation time after which the topography should stop
            increasing. If not specified, a time-invariant terrain surface-height
            is assumed.
        smooth : bool
            ``True`` to smooth the topography out, ``False`` otherwise.
        **kwargs :
            Keyword arguments for the exclusive use of the subclass.
        """
        self._type = None

        # get the steady profile
        topo_steady = self.compute_steady_profile(grid, **kwargs)

        # smooth the topography out
        if smooth:
            topo_steady[1:-1, 1:-1] += 0.125 * (
                topo_steady[:-2, 1:-1]
                + topo_steady[2:, 1:-1]
                + topo_steady[1:-1, :-2]
                + topo_steady[1:-1, 2:]
                - 4.0 * topo_steady[1:-1, 1:-1]
            )

        # wrap the steady topography profile in a DataArray
        topo_steady = DataArray(
            topo_steady,
            coords=[grid.x.values, grid.y.values],
            dims=[grid.x.dims[0], grid.y.dims[0]],
            attrs={"units": "m"},
        )

        super().__init__(topo_steady, time=time)

    @property
    def type(self) -> str:
        """ String used to register the subclass. """
        assert self._type is not None
        return self._type

    @type.setter
    def type(self, topography_type) -> None:
        self._type = topography_type

    @abc.abstractmethod
    def compute_steady_profile(
        self, grid: "PhysicalHorizontalGrid", **kwargs
    ) -> np.ndarray:
        """ Compute the steady topography profile.

        Parameters
        ----------
        grid : tasmania.PhysicalHorizontalGrid
            The underlying :class:`~tasmania.PhysicalHorizontalGrid`.
        **kwargs :
            The keyword arguments passed to the constructor.

        Returns
        -------
        np.ndarray :
            The steady topography profile in [m].
        """
        pass

    @staticmethod
    def factory(
        topography_type: str,
        grid: "PhysicalHorizontalGrid",
        time: Optional[taz_types.timedelta_t] = None,
        smooth: bool = False,
        **kwargs
    ):
        """ Get an instance of a derived class.

        Parameters
        ----------
        topography_type : str
            The topography type, i.e. the string used to register the subclass
            which should be instantiated.
        time : `datetime.timedelta`, optional
            The elapsed simulation time after which the topography should stop
            increasing. If not specified, a time-invariant terrain surface-height
            is assumed.
        smooth : `bool`, optional
            ``True`` to smooth the topography out, ``False`` otherwise.
        **kwargs :
            Keyword arguments for the exclusive use of the subclass.

        Returns
        -------
        obj :
            An instance of the derived class registered as ``topography_type``.
        """
        args = (grid, time, smooth)

        obj = factorize(topography_type, PhysicalTopography, args, kwargs)
        obj.type = topography_type

        return obj


class NumericalTopography(Topography):
    """ A time-dependent topography defined over a numerical grid. """

    def __init__(
        self,
        grid: "NumericalHorizontalGrid",
        phys_topography: PhysicalTopography,
        boundary: "HorizontalBoundary",
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.NumericalHorizontalGrid
            The underlying :class:`~tasmania.NumericalHorizontalGrid`.
        phys_topography : tasmania.Topography
            The topography defined over the associated physical grid.
        boundary : tasmania.HorizontalBoundary
            The :class:`~tasmania.HorizontalBoundary` handling the horizontal
            boundary conditions.
        """
        self._type = phys_topography.type
        topo_time = phys_topography.time

        ptopo = phys_topography.profile.values
        ptopo_steady = phys_topography.steady_profile.values
        units = phys_topography.profile.attrs["units"]

        ctopo = get_dataarray_2d(boundary.get_numerical_field(ptopo), grid, units)
        ctopo_steady = get_dataarray_2d(
            boundary.get_numerical_field(ptopo_steady), grid, units
        )

        super().__init__(ctopo_steady, ctopo, topo_time)

    @property
    def type(self) -> str:
        """ The topography type. """
        return self._type
