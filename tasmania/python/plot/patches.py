# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from typing import Optional

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.plot_utils import (
    add_annotation,
    make_circle,
    make_lineplot,
    make_rectangle,
)
from tasmania.python.utils import typingx


class Annotation(Drawer):
    """
    Add an annotation.
    """

    def __init__(
        self, properties: Optional[typingx.options_dict_t] = None
    ) -> None:
        """
        Parameters
        ----------
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            properties, and whose values specify values for those properties.
            See :func:`~tasmania.python.plot.plot_utils.add_annotation`.
        """
        super().__init__(properties)

    def __call__(
        self, state: typingx.DataArrayDict, fig: plt.Figure, ax: plt.Axes
    ) -> None:
        add_annotation(ax, **self.properties)


class Circle(Drawer):
    """
    Drawer plotting a circle.
    """

    def __init__(self, properties: Optional[typingx.options_dict_t] = None):
        """
        Parameters
        ----------
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            properties, and whose values specify values for those properties.
            See :func:`~tasmania.python.plot.plot_utils.make_circle`.
        """
        super().__init__(properties)

    def __call__(
        self, state: typingx.DataArrayDict, fig: plt.Figure, ax: plt.Axes
    ) -> None:
        make_circle(ax, **self.properties)


class Rectangle(Drawer):
    """
    Drawer plotting a rectangle.
    """

    def __init__(self, properties: Optional[typingx.options_dict_t] = None):
        """
        Parameters
        ----------
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            settings, and whose values specify values for those settings.
            See :func:`~tasmania.python.plot.plot_utils.make_rectangle`.
        """
        super().__init__(properties)

    def __call__(
        self, state: typingx.DataArrayDict, fig: plt.Figure, ax: plt.Axes
    ) -> None:
        make_rectangle(ax, **self.properties)


class Segment(Drawer):
    """
    Drawer plotting a segment.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        properties: Optional[typingx.options_dict_t] = None,
    ) -> None:
        """
        Parameters
        ----------
        x : array
            2-items :class:`numpy.array` storing the x-coordinates of the
            line end-points.
        y : array
            2-items :class:`numpy.array` storing the y-coordinates of the
            line end-points.
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            properties, and whose values specify values for those properties.
            See :func:`~tasmania.python.plot.plot_utils.make_lineplot`.
        """
        self.x, self.y = x, y
        super().__init__(properties)

    def __call__(
        self, state: typingx.DataArrayDict, fig: plt.Figure, ax: plt.Axes
    ) -> None:
        make_lineplot(
            np.array(self.x), np.array(self.y), ax, **self.properties
        )
