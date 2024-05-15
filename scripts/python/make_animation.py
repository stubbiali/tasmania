# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
import tasmania as taz


class AnimationWrapper:
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            wrapper_module = data["artist"]["wrapper_module"]
            wrapper_classname = data["artist"]["wrapper_classname"]
            wrapper_config = data["artist"]["wrapper_config"]

            import_str = "from {} import {}".format(
                wrapper_module, wrapper_classname
            )
            exec(import_str)
            self.artist_wrapper = locals()[wrapper_classname](wrapper_config)

            fps = data["fps"]

            self.core = taz.Animation(
                self.artist_wrapper.get_artist(), fps=fps
            )

            self.tlevels = range(
                data["tlevels"][0], data["tlevels"][1], data["tlevels"][2]
            )
            self.save_dest = data["save_dest"]

    def store(self) -> None:
        for tlevel in self.tlevels:
            states = self.artist_wrapper.get_states(tlevel)
            self.core.store(*states)

    def run(self) -> None:
        self.core.run(save_dest=self.save_dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a figure with a single plot."
    )
    parser.add_argument(
        "configfile",
        metavar="configfile",
        type=str,
        help="JSON configuration file.",
    )
    args = parser.parse_args()
    animation = AnimationWrapper(args.configfile)
    animation.store()
    animation.run()
