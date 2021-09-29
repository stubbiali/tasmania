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
import json
import numpy as np

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.storage import get_dataarray_3d

from scripts.python.data_loaders.base import BaseLoader
from scripts.python.data_loaders.mounter import DatasetMounter


class DomainCumulativeLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            filename = "".join(data["filename"])
            self.dsmounter = DatasetMounter(filename)

            self.fname = data["field_name"]
            self.funits = data["field_units"]

            start, stop, step = data["xslice"]
            self.xslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["yslice"]
            self.yslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["zslice"]
            self.zslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        g = self.dsmounter.get_grid()
        nx, ny = g.nx, g.ny
        x, y, z = self.xslice, self.yslice, self.zslice

        state = self.dsmounter.get_state(tlevel)
        field = state[self.fname].to_units(self.funits).values[x, y, z]
        state["domain_cumulative_" + self.fname] = get_dataarray_3d(
            np.sum(np.sum(np.sum(field, axis=2), axis=1), axis=0)
            * np.ones((nx, ny, 1)),
            g,
            self.funits,
        )

        return state


class ColumnCumulativeLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            filename = "".join(data["filename"])
            self.dsmounter = DatasetMounter(filename)

            self.fname = data["field_name"]
            self.funits = data["field_units"]

            start, stop, step = data["xslice"]
            self.xslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["yslice"]
            self.yslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["zslice"]
            self.zslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        g = self.dsmounter.get_grid()
        x, y, z = self.xslice, self.yslice, self.zslice

        state = self.dsmounter.get_state(tlevel)
        field = state[self.fname].to_units(self.funits).values[x, y, z]
        state["column_cumulative_" + self.fname] = get_dataarray_3d(
            np.sum(field, axis=2)[:, :, np.newaxis], g, self.funits
        )

        return state


class TotalPrecipitationLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            filename = "".join(data["filename"])
            self.dsmounter = DatasetMounter(filename)

            start, stop, step = data["xslice"]
            self.xslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["yslice"]
            self.yslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["zslice"]
            self.zslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        g = self.dsmounter.get_grid()
        nx, ny = g.nx, g.ny
        x, y, z = self.xslice, self.yslice, self.zslice

        state = self.dsmounter.get_state(tlevel)
        field = state["precipitation"].to_units("m hr^-1").values[x, y, z]
        dx = g.dx.to_units("m").values.item()
        dy = g.dx.to_units("m").values.item()
        state["domain_cumulative_precipitation"] = get_dataarray_3d(
            1000
            * dx
            * dy
            * np.sum(np.sum(np.sum(field, axis=2), axis=1), axis=0)
            * np.ones((nx, ny, 1)),
            g,
            "kg hr^-1",
        )

        return state


class TotalAccumulatedPrecipitationLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            filename = "".join(data["filename"])
            self.dsmounter = DatasetMounter(filename)

            start, stop, step = data["xslice"]
            self.xslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )
            start, stop, step = data["yslice"]
            self.yslice = (
                None
                if start == stop == step is None
                else slice(start, stop, step)
            )

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        g = self.dsmounter.get_grid()
        dx, dy = (
            g.dx.to_units("m").values.item(),
            g.dy.to_units("m").values.item(),
        )
        nx, ny = g.nx, g.ny
        x, y = self.xslice, self.yslice

        state = self.dsmounter.get_state(tlevel)
        accprec = (
            state["accumulated_precipitation"].to_units("mm").values[x, y, 0]
        )
        state["total_accumulated_precipitation"] = get_dataarray_3d(
            dx
            * dy
            * np.sum(np.sum(accprec, axis=1), axis=0)
            * np.ones((nx, ny, 1)),
            g,
            "kg",
        )

        return state
