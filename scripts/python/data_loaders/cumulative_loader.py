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
import json
import numpy as np
import tasmania as taz
if __name__ == '__main__':
        from base_loader import BaseLoader
        from mounter import DatasetMounter
else:
        from .base_loader import BaseLoader
        from .mounter import DatasetMounter


class DomainCumulativeLoader(BaseLoader):
        def __init__(self, json_filename):
                with open(json_filename, 'r') as json_file:
                        data = json.load(json_file)
                        filename = ''.join(data['filename'])
                        self._dsmounter = DatasetMounter(filename)

                        self._fname = data['field_name']
                        self._funits = data['field_units']

                        start, stop, step = data['xslice']
                        self._xslice = None if start == stop == step is None else slice(start, stop, step)
                        start, stop, step = data['yslice']
                        self._yslice = None if start == stop == step is None else slice(start, stop, step)
                        start, stop, step = data['zslice']
                        self._zslice = None if start == stop == step is None else slice(start, stop, step)

        def get_grid(self):
                return self._dsmounter.get_grid()

        def get_nt(self):
                return self._dsmounter.get_nt()

        def get_initial_time(self):
                return self._dsmounter.get_state(0)['time']

        def get_state(self, tlevel):
                g = self._dsmounter.get_grid()
                nx, ny = g.nx, g.ny
                x, y, z = self._xslice, self._yslice, self._zslice

                state = self._dsmounter.get_state(tlevel)
                field = state[self._fname].to_units(self._funits).values[x, y, z]
                state['domain_cumulative_' + self._fname] = taz.make_dataarray_3d(
                        np.sum(np.sum(np.sum(field, axis=2), axis=1), axis=0) * np.ones((nx, ny, 1)),
                        g, self._funits
                )

                return state


class ColumnCumulativeLoader(BaseLoader):
        def __init__(self, json_filename):
                with open(json_filename, 'r') as json_file:
                        data = json.load(json_file)
                        filename = ''.join(data['filename'])
                        self._dsmounter = DatasetMounter(filename)

                        self._fname = data['field_name']
                        self._funits = data['field_units']

                        start, stop, step = data['zslice']
                        self._zslice = None if start == stop == step is None else slice(start, stop, step)

        def get_grid(self):
                return self._dsmounter.get_grid()

        def get_nt(self):
                return self._dsmounter.get_nt()

        def get_initial_time(self):
                return self._dsmounter.get_state(0)['time']

        def get_state(self, tlevel):
                g = self._dsmounter.get_grid()
                nx, ny = g.nx, g.ny
                z = self._zslice

                state = self._dsmounter.get_state(tlevel)
                field = state[self._fname].to_units(self._funits).values[:, :, z]
                state['column_cumulative_' + self._fname] = taz.make_dataarray_3d(
                        np.sum(field, axis=2)[:, :, np.newaxis], g, self._funits
                )

                return state


class TotalAccumulatedPrecipitationLoader(BaseLoader):
        def __init__(self, json_filename):
                with open(json_filename, 'r') as json_file:
                        data = json.load(json_file)
                        filename = ''.join(data['filename'])
                        self._dsmounter = DatasetMounter(filename)

                        start, stop, step = data['xslice']
                        self._xslice = None if start == stop == step is None else slice(start, stop, step)
                        start, stop, step = data['yslice']
                        self._yslice = None if start == stop == step is None else slice(start, stop, step)

        def get_grid(self):
                return self._dsmounter.get_grid()

        def get_nt(self):
                return self._dsmounter.get_nt()

        def get_initial_time(self):
                return self._dsmounter.get_state(0)['time']

        def get_state(self, tlevel):
                g = self._dsmounter.get_grid()
                dx, dy = g.dx.to_units('m').values.item(), g.dy.to_units('m').values.item()
                nx, ny = g.nx, g.ny
                x, y = self._xslice, self._yslice

                state = self._dsmounter.get_state(tlevel)
                accprec = state['accumulated_precipitation'].to_units('mm').values[x, y, 0]
                state['total_accumulated_precipitation'] = taz.make_dataarray_3d(
                        dx * dy * np.sum(np.sum(accprec, axis=1), axis=0) * np.ones((nx, ny, 1)),
                        g, 'kg'
                )

                return state

