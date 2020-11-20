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
from collections import UserDict, abc
import re
from typing import Sequence, Union

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import ProtocolError


class Registry(UserDict):
    """A dict-like registry."""

    def find_key(self, target: str):
        if target in self.keys():
            return target
        for key in self.keys():
            pattern = re.compile(rf"{key}")
            if pattern.match(target):
                return key
        if prt.catch_all in self.keys():
            return prt.catch_all

        raise KeyError(f"Key '{target}' not found.")

    def __getitem__(self, key: Union[str, Sequence[str]]):
        if not (isinstance(key, str) or isinstance(key, abc.Sequence)):
            raise TypeError("key must be either a str or a sequence of str.")

        keys = (key,) if isinstance(key, str) else key
        target = key if isinstance(key, str) else key[0]

        if len(keys) == 1:
            return super().__getitem__(self.find_key(target))
        else:
            return (
                super().__getitem__(self.find_key(target)).__getitem__(key[1:])
            )

    def __setitem__(self, key, value):
        if isinstance(key, str) or (
            isinstance(key, abc.Sequence) and len(key) == 1
        ):
            key = key if isinstance(key, str) else key[0]
            return super().__setitem__(key, value)
        elif isinstance(key, Sequence):
            subdict = self.setdefault(key[0], Registry())
            return subdict.__setitem__(key[1:], value)
        else:
            raise TypeError("key must be either a str or a sequence of str.")


def filter_args_list(args):
    out = []
    unset_attributes = set(prt.attribute_names)

    # get attributes which have not been set
    for i in range(0, len(args), 2):
        if args[i] in unset_attributes:
            if args[i] == prt.master_attribute:
                if args[i + 1] not in prt.master_attribute_values:
                    raise ProtocolError(
                        f"Unknown value '{args[i+1]}' for master "
                        f"attribute '{args[i]}'."
                    )
            unset_attributes.remove(args[i])
            out.append(args[i])
            out.append(args[i + 1])
        else:
            raise ProtocolError(f"Unknown protocol attribute '{args[i]}'.")

    # set the unset attributes...
    for attribute in unset_attributes:
        # ... provided that they are defaulted
        if attribute not in prt.attribute_defaults:
            raise ProtocolError(
                f"The non-default protocol attribute '{attribute}' has "
                f"not been provided."
            )
        out.append(attribute)
        out.append(prt.attribute_defaults[attribute])

    return out


def set_protocol_attributes(handle, args):
    for i in range(0, len(args), 2):
        if getattr(handle, args[i], args[i + 1]) is not args[i + 1]:
            raise ProtocolError(
                f"Name conflict: Object '{handle.__name__}' already "
                f"sets the attribute '{args[i]}' to "
                f"'{getattr(handle, args[i])}'."
            )

        try:
            setattr(handle, args[i], args[i + 1])
        except AttributeError:
            # bound attributes do not define the __dict__ attribute
            handle.__dict__[args[i]] = args[i + 1]

    return handle


def register(registry, *args):
    # filter arguments list
    args_list = filter_args_list(args)

    def core(handle):
        # add attributes to function
        set_protocol_attributes(handle, args_list)

        # insert function into registry
        keys = tuple(args_list[i] for i in range(1, len(args_list), 2))
        registry[keys] = handle

        return handle

    return core
