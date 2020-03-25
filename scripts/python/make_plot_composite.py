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
from datetime import datetime
import json
from matplotlib import pyplot as plt
from tasmania import PlotComposite, taz_types
from tasmania.python.utils.utils import assert_sequence
from typing import List, Optional, Sequence, Tuple, Union


class PlotCompositeWrapper:
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            self.plot_wrappers = []

            for plot_data in data["plots"]:
                wrapper_module = plot_data["wrapper_module"]
                wrapper_classname = plot_data["wrapper_classname"]
                wrapper_config = plot_data["wrapper_config"]

                import_str = "from {} import {}".format(wrapper_module, wrapper_classname)
                exec(import_str)
                self.plot_wrappers.append(locals()[wrapper_classname](wrapper_config))

            nrows = data["nrows"]
            ncols = data["ncols"]

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

            self.core = PlotComposite(
                *(wrapper.get_artist() for wrapper in self.plot_wrappers),
                nrows=nrows,
                ncols=ncols,
                interactive=False,
                print_time=print_time,
                init_time=init_time,
                figure_properties=figure_properties
            )

            self.tlevels = data.get("tlevels", (0,) * len(self.plot_wrappers))
            self.save_dest = data.get("save_dest", None)

    def get_artist(self) -> PlotComposite:
        return self.core

    def get_states(
        self, tlevels: Optional[Union[int, Sequence[int]]] = None
    ) -> Tuple[Union[taz_types.dataarray_dict_t, Sequence[taz_types.dataarray_dict_t]]]:
        tlevels = self.tlevels if tlevels is None else tlevels
        tlevels = (
            (tlevels,) * len(self.plot_wrappers) if isinstance(tlevels, int) else tlevels
        )
        assert_sequence(tlevels, reflen=len(self.plot_wrappers), reftype=int)

        states = tuple(
            plot_wrapper.get_states(tls)
            for plot_wrapper, tls in zip(self.plot_wrappers, tlevels)
        )

        return states

    def store(
        self,
        tlevels: Optional[Union[int, Sequence[int]]] = None,
        fig: Optional[plt.Figure] = None,
        show: bool = False,
    ) -> plt.Figure:
        states = self.get_states(tlevels)
        return self.core.store(*states, fig=fig, save_dest=self.save_dest, show=show)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a figure with multiple panels."
    )
    parser.add_argument(
        "configfile", metavar="configfile", type=str, help="JSON configuration file."
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_const",
        const=0,
        default=1,
        help="Do not show the generated plot.",
    )
    args = parser.parse_args()
    plot_wrapper = PlotCompositeWrapper(args.configfile)
    plot_wrapper.store(show=args.show)
