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
import click
from datetime import datetime
import json
from matplotlib import pyplot as plt
from typing import List, Optional, Sequence, Tuple, Union

from tasmania.python.plot.monitors import Plot
from tasmania.python.utils import typingx as ty


class PlotWrapper:
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            self.drawer_wrappers = []

            for drawer_data in data["drawers"]:
                loader_module = drawer_data["loader_module"]
                loader_classname = drawer_data["loader_classname"]
                loader_config = drawer_data["loader_config"]

                import_str = "from {} import {}".format(
                    loader_module, loader_classname
                )
                exec(import_str)
                loader = locals()[loader_classname](loader_config)

                wrapper_module = drawer_data["wrapper_module"]
                wrapper_classname = drawer_data["wrapper_classname"]
                wrapper_config = drawer_data["wrapper_config"]

                import_str = "from {} import {}".format(
                    wrapper_module, wrapper_classname
                )
                exec(import_str)
                self.drawer_wrappers.append(
                    locals()[wrapper_classname](loader, wrapper_config)
                )

            print_time = data["print_time"]

            if print_time == "elapsed" and "init_time" in data:
                init_time = datetime(
                    year=data["init_time"]["year"],
                    month=data["init_time"]["month"],
                    day=data["init_time"]["day"],
                    hour=data["init_time"].get("hour", 0),
                    minute=data["init_time"].get("minute", 0),
                    second=data["init_time"].get("second", 0),
                )
            else:
                init_time = None

            figure_properties = data["figure_properties"]
            self.axes_properties = data["axes_properties"]

            self.core = Plot(
                *(wrapper.get_drawer() for wrapper in self.drawer_wrappers),
                interactive=False,
                print_time=print_time,
                init_time=init_time,
                figure_properties=figure_properties,
                axes_properties=self.axes_properties
            )

            self.tlevels = data.get("tlevels", 0)
            self.save_dest = data.get("save_dest", None)

    def get_artist(self) -> Plot:
        return self.core

    def get_states(
        self, tlevels: Optional[Union[int, Sequence[int]]] = None
    ) -> List[ty.DataArrayDict]:
        wrappers = self.drawer_wrappers

        tlevels = self.tlevels if tlevels is None else tlevels
        tlevels = (
            (tlevels,) * len(wrappers) if isinstance(tlevels, int) else tlevels
        )

        states = []

        for wrapper, tlevel in zip(wrappers, tlevels):
            states.append(wrapper.get_state(tlevel))

        return states

    def store(
        self,
        tlevels: Optional[Union[int, Sequence[int]]] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        states = self.get_states(tlevels)
        return self.core.store(
            *states, fig=fig, ax=ax, save_dest=self.save_dest, show=show
        )


@click.command()
@click.option(
    "-j", "--jsonfile", type=str, default=None, help="JSON configuration file."
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Disable visualization of generated figure.",
)
def main(jsonfile, no_show=False):
    plot_wrapper = PlotWrapper(jsonfile)
    plot_wrapper.store(show=not no_show)


if __name__ == "__main__":
    main()
