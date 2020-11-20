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
# magic string which can be assigned to a attribute of the protocol to indicate
# that the behaviour of that attribute is the same irrespective of the value
# assumed by the other protocol attributes
catch_all = "all"

# the attributes of the protocol
# attributes are listed in a dict where keys represent the names of the
# attributes, and values represent either labels, tags or purposes of the
# attributes
attributes = {
    "__backend__": "backend",
    "__functionality__": "functionality",
    "__stencil__": "stencil",
}

# the list of the names of the attributes
attribute_names = attributes.keys()

# default value for each protocol attribute
# if a attribute does not appear in the dict, it means that that attribute is
# not default, i.e. it must be explicitly specified
attribute_defaults = {"__stencil__": catch_all}

# the hierarchy of attributes
# this sets the order in which protocol attributes are looked-up when comparing
# two objects which implement the protocol
attributes_hierarchy = ("__functionality__", "__backend__", "__stencil__")

# the master attribute, i.e. the first item of the hierarchy of attributes
master_attribute = attributes_hierarchy[0]

# the values which the master attribute may take
master_attribute_values = ("compiler", "definition", "empty", "ones", "zeros")
