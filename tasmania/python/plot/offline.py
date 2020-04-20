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
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Sequence, TYPE_CHECKING, Union

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.plot_utils import make_lineplot
from tasmania.python.utils import taz_types

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


class Line(Drawer):
    """
    Draw a line by retrieving a scalar value from multiple states which might
    be defined over different domain.
    """

    def __init__(
        self,
        grids: "Sequence[Grid]",
        field_name: str,
        field_units: str,
        x: Union[int, Sequence[int]],
        y: Union[int, Sequence[int]],
        z: Union[int, Sequence[int]],
        xdata: Optional[np.ndarray] = None,
        ydata: Optional[np.ndarray] = None,
        properties: Optional[taz_types.options_dict_t] = None,
    ) -> None:
        """
        Parameters
        ----------
        grids : Sequence[tasmania.Grid]
            The :class:`tasmania.Grid`s underlying the states.
        field_name : str
            The state quantity to visualize.
        field_units : str
            The units for the quantity to visualize.
        x : `int` or `tuple[int]`
            For each state, the index along the first dimension of the field array
            identifying the grid point to consider. If the same index applies to
            all states, it can be specified as an integer.
        y : `int` or `tuple[int]`
            For each state, the index along the second dimension of the field array
            identifying the grid point to consider. If the same index applies to
            all states, it can be specified as an integer.
        z : `int` or `tuple[int]`
            For each state, the index along the third dimension of the field array
            identifying the grid point to consider. If the same index applies to
            all states, it can be specified as an integer.
        xdata : `np.ndarray`, optional
            The data to be placed on the horizontal axis of the plot. If specified,
            the data retrieved from the states will be placed on the vertical axis
            of the plot. Only allowed if ``ydata`` is not given.
        ydata : `np.ndarray`, optional
            The data to be placed on the vertical axis of the plot. If specified,
            the data retrieved from the states will be placed on the horizontal axis
            of the plot. Only allowed if ``xdata`` is not given.
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            settings, and whose values specify values for those settings.
            See :func:`tasmania.python.plot.utils.make_lineplot`.
        """
        super().__init__(properties)

        x = [x] * len(grids) if isinstance(x, int) else x
        y = [y] * len(grids) if isinstance(y, int) else y
        z = [z] * len(grids) if isinstance(z, int) else z

        self._retrievers = []
        for k in range(len(grids)):
            slice_x = slice(x[k], x[k] + 1 if x[k] != -1 else None)
            slice_y = slice(y[k], y[k] + 1 if y[k] != -1 else None)
            slice_z = slice(z[k], z[k] + 1 if z[k] != -1 else None)
            self._retrievers.append(
                DataRetriever(
                    grids[k], field_name, field_units, slice_x, slice_y, slice_z
                )
            )

        assert (
            xdata is None or ydata is None
        ), "Both xdata and ydata are given, but only one is allowed."

        if xdata is not None:
            self._xdata = xdata
            self._ydata = []
            self._data_on_yaxis = True
        else:
            self._xdata = []
            self._ydata = ydata
            self._data_on_yaxis = False

    def reset(self) -> None:
        if self._data_on_yaxis:
            self._ydata = []
        else:
            self._xdata = []

    def __call__(
        self,
        state: taz_types.dataarray_dict_t,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        if self._data_on_yaxis:
            k = len(self._ydata)
            if k >= len(self._retrievers):
                raise RuntimeError(
                    "You exceeded the maximum number of states ({}) which could be "
                    "passed between two consecutive calls to reset().".format(
                        len(self._retrievers)
                    )
                )
            self._ydata.append(self._retrievers[k](state))
        else:
            k = len(self._xdata)
            if k >= len(self._retrievers):
                raise RuntimeError(
                    "You exceeded the maximum number of states ({}) which could be "
                    "passed between two consecutive calls to reset().".format(
                        len(self._retrievers)
                    )
                )
            self._xdata.append(self._retrievers[k](state))

        if ax is not None:
            k = len(self._ydata) if self._data_on_yaxis else len(self._xdata)
            make_lineplot(
                np.squeeze(np.array(self._xdata[:k])),
                np.squeeze(np.array(self._ydata[:k])),
                ax,
                **self.properties
            )
