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
from tasmania.python.framework.promoters import (
    Diagnostic2Tendency,
    Tendency2Diagnostic,
)
from tasmania.python.utils import typing


class AirPotentialTemperature2Diagnostic(Tendency2Diagnostic):
    """ Promoting the tendency of air potential temperature to state variable. """

    @property
    def input_properties(self) -> typing.PropertiesDict:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_potential_temperature": {
                "dims": dims,
                "units": "K s^-1",
                "diagnostic_name": "tendency_of_air_potential_temperature",
                "remove_from_tendencies": True,
            }
        }

        return return_dict


class AirPotentialTemperature2Tendency(Diagnostic2Tendency):
    """ Downgrading the tendency of air potential temperature to tendency variable. """

    @property
    def input_properties(self) -> typing.PropertiesDict:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "tendency_of_air_potential_temperature": {
                "dims": dims,
                "units": "K s^-1",
                "tendency_name": "air_potential_temperature",
                "remove_from_diagnostics": False,
            }
        }

        return return_dict
