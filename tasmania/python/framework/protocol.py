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
# magic string which can be assigned to a key of the protocol dict to indicate
# that the value of that key is the same irrespective of the value
# assumed by the other keys
wildcard = "all"

# the attribute which an object should define to implement the protocol
attribute = "__tasmania__"

# the protocol attribute should be a dictionary containing the following keys
keys = ("backend", "function", "stencil")

# default value for each key
# if a key does not appear in the dict, it means that that key is
# not default, i.e. it must be explicitly specified
defaults = {"stencil": wildcard}

# the hierarchy of the keys
# this sets the order in which keys are looked-up when comparing
# two objects which implement the protocol
keys_hierarchy = ("function", "backend", "stencil")

# the master key, i.e. the first item of the hierarchy of keys
master_key = keys_hierarchy[0]

# the values which the master key may take
master_key_values = (
    "stencil_compiler",
    "stencil_definition",
    "empty",
    "ones",
    "zeros",
)

# the name of the dictionary which stores the values taken by the protocol
# keys at runtime
runtime_attribute = "__tasmania_runtime__"
