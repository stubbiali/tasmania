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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains
"""
import time


class TimeMeasurement:
    def __init__(self, name="Default time measurement", level=1):
        self.tmname = name
        self.tmstart = None
        self.tmstop = None
        self.tmtotal = 0.0
        self.tmlevel = level

class Timings:
    def __init__(self, name="Overall running time"):
        self.tname = name
        overall_measurement = TimeMeasurement(name, level=0)
        self.timing_list = [overall_measurement, ]

    def start(self, name, level):
        started = False
        for item in self.timing_list:
            if item.tmname == name:
                item.tmtotal += item.tmstop - item.tmstart
                item.tmstop = None
                item.tmstart = time.perf_counter()
                started = True
                break
        if not started:
            tm = TimeMeasurement(name=name, level=level)
            tm.tmstart = time.perf_counter()
            self.timing_list.append(tm)

    def stop(self, name):
        ctime = time.perf_counter()
        ended = False
        for item in self.timing_list:
            if item.tmname == name:
                if item.tmstop is None:
                    item.tmstop = ctime
                    ended = True
                    break
                else:
                    raise ValueError("Timer.py: Timings.stop(name={}) called"
                                     " for already stopped timing.".format(name))

        if not ended:
            raise ValueError("Timer.py: Timings.stop(name={} called"
                             " for a timing that has not been started.".format(name))

    def list_timings(self):
        for item in self.timing_list:
            tree_string = (item.tmlevel - 1) * "-"
            if item.tmlevel == 0:
                print("{0:50}".format(item.tmname))
            elif item.tmstart is not None and item.tmstop is not None:
                print("{0:50} T = {1: 12.4}".format(tree_string + item.tmname, item.tmtotal
                                                    + (item.tmstop - item.tmstart)))
            else:
                raise ValueError("Timer.py: Timings.list_timings() called"
                                 " with at least one time measurement never stopped.")
