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
from tasmania.python.framework.promoter import (
    FromDiagnosticToTendency,
    FromTendencyToDiagnostic,
)


class AirPotentialTemperatureToDiagnostic(FromTendencyToDiagnostic):
    """Promoting the tendency of air potential temperature to state variable."""

    @property
    def input_tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_potential_temperature": {
                "dims": dims,
                "units": "K s^-1",
                "diagnostic_name": "tendency_of_air_potential_temperature",
            }
        }

        return return_dict


class AirPotentialTemperatureToTendency(FromDiagnosticToTendency):
    """Downgrading the tendency of air potential temperature to tendency variable."""

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "tendency_of_air_potential_temperature": {
                "dims": dims,
                "units": "K s^-1",
                "tendency_name": "air_potential_temperature",
            }
        }

        return return_dict
