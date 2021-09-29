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
from tasmania import Timer

if __name__ == "__main__":
    for _ in range(10):
        Timer.start(label="level-0-0")
        Timer.start(label="level-1-0")
        Timer.start(label="level-2-0")
        Timer.stop()
        Timer.start(label="level-2-1")
        Timer.start(label="level-3-0")
        Timer.stop()
        Timer.stop()
        Timer.start(label="level-2-2")
        Timer.stop()
        Timer.stop()
        Timer.stop()

    Timer.log()
