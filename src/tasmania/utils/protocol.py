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

from __future__ import annotations
from collections import UserDict, abc
import itertools
import re
from typing import TYPE_CHECKING

from tasmania.framework import protocol as prt
from tasmania.utils.exceptions import ProtocolError

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence, Union


class Registry(UserDict):
    """A dict-like registry."""

    def find_key(self, target: str) -> str:
        if target in self.keys():
            return target
        for key in self.keys():
            pattern = re.compile(rf"{key}")
            if pattern.match(target):
                return key
        if prt.wildcard in self.keys():
            return prt.wildcard

        raise KeyError(f"Key '{target}' not found.")

    def __getitem__(self, key: Union[str, Sequence[str]]) -> Any:
        if not (isinstance(key, str) or isinstance(key, abc.Sequence)):
            raise TypeError("key must be either a str or a sequence of str.")

        keys = (key,) if isinstance(key, str) else key
        target = key if isinstance(key, str) else key[0]

        if len(keys) == 1:
            return super().__getitem__(self.find_key(target))
        else:
            return super().__getitem__(self.find_key(target)).__getitem__(key[1:])

    def __setitem__(self, key: Union[str, Sequence[str]], value: Any) -> None:
        if isinstance(key, str) or (isinstance(key, abc.Sequence) and len(key) == 1):
            key = key if isinstance(key, str) else key[0]
            return super().__setitem__(key, value)
        elif isinstance(key, Sequence):
            subdict = self.setdefault(key[0], Registry())
            return subdict.__setitem__(key[1:], value)
        else:
            raise TypeError("key must be either a str or a sequence of str.")


def filter_args_list(args: Sequence[str]) -> list[str]:
    out = []
    unset_keys = set(prt.keys)

    # get keys which have not been set
    for i in range(0, len(args), 2):
        if args[i] in unset_keys:
            if args[i] == prt.master_key:
                if args[i + 1] not in prt.master_key_values:
                    raise ProtocolError(
                        f"Unknown value '{args[i+1]}' for the master "
                        f"key '{args[i]}' of the '{prt.attribute}' dictionary."
                    )
            unset_keys.remove(args[i])
            out.append(args[i])
            out.append(args[i + 1])
        else:
            raise ProtocolError(f"Unknown key '{args[i]}' in the '{prt.attribute}' dictionary.")

    # set the unset keys...
    for key in unset_keys:
        # ... provided that they are defaulted
        if key not in prt.defaults:
            raise ProtocolError(
                f"The non-default key '{key}' of the '{prt.attribute}' "
                f"dictionary has not been provided."
            )
        out.append(key)
        out.append(prt.defaults[key])

    # assess that the keys hierarchy is respected
    for i in range(0, len(out), 2):
        if out[i] != prt.keys_hierarchy[i // 2]:
            raise ProtocolError(f"The key '{out[i]}' does not adhere to the keys hierarchy.")

    return out


def set_attribute(handle: Callable, *args: str) -> Callable:
    try:
        d = getattr(handle, prt.attribute, None)
    except AttributeError:
        # bound attributes do not define the __dict__ attribute
        d = handle.__dict__.get(prt.attribute, None)

    if d is not None:
        if not isinstance(d, dict):
            raise ProtocolError(
                f"The object '{handle.__name__}' already defines the attribute '{prt.attribute}' "
                f"as a non-dict object."
            )
    else:
        d = {}

    for i in range(0, len(args), 2):
        if args[i] in d:
            if isinstance(d[args[i]], str) and d[args[i]] != args[i + 1]:
                d[args[i]] = [d[args[i]], args[i + 1]]
            elif isinstance(d[args[i]], abc.Sequence) and args[i + 1] not in d[args[i]]:
                d[args[i]].append(args[i + 1])
            else:
                d[args[i]] = args[i + 1]
        else:
            d[args[i]] = args[i + 1]

    try:
        setattr(handle, prt.attribute, d)
    except AttributeError:
        # bound attributes do not define the __dict__ attribute
        handle.__dict__[prt.attribute] = d

    return handle


def set_runtime_attribute(handle: Callable, *args: str) -> Callable:
    try:
        d = getattr(handle, prt.runtime_attribute, None)
    except AttributeError:
        # bound attributes do not define the __dict__ attribute
        d = handle.__dict__.get(prt.runtime_attribute, None)

    if d is not None:
        if not isinstance(d, dict):
            raise ProtocolError(
                f"The object '{handle.__name__}' already defines the attribute "
                f"'{prt.runtime_attribute}' as a non-dict object."
            )
    else:
        d = {}

    for i in range(0, len(args), 2):
        d[args[i]] = args[i + 1]

    try:
        setattr(handle, prt.runtime_attribute, d)
    except AttributeError:
        # bound attributes do not define the __dict__ attribute
        handle.__dict__[prt.runtime_attribute] = d

    return handle


def singleregister(
    handle: Optional[Callable] = None,
    registry: Optional[Registry] = None,
    args: Optional[Sequence[Union[str, Sequence[str]]]] = None,
) -> Callable:
    # filter arguments list
    args_list = filter_args_list(args)

    def core(func):
        # add attributes to function
        set_attribute(func, *args_list)
        set_runtime_attribute(func)

        if registry is not None:
            # insert function into registry
            keys = tuple(args_list[i] for i in range(1, len(args_list), 2))
            registry[keys] = func

        return func

    if handle is None:
        return core
    else:
        return core(handle)


def multiregister(
    handle: Optional[Callable] = None,
    registry: Optional[Registry] = None,
    args: Optional[Sequence[Union[str, Sequence[str]]]] = None,
) -> Callable:
    new_args_list = []
    for arg in args:
        new_arg = (arg,) if isinstance(arg, str) else arg
        new_args_list.append(new_arg)

    decorators = []
    for new_args in itertools.product(*new_args_list):
        tmp = singleregister(handle, registry, new_args)
        if handle is None:
            decorators.append(tmp)

    def core(func):
        for decorator in decorators:
            decorator(func)

        return func

    if handle is None:
        return core
    else:
        return handle


def add_method(obj, method_name, method_handle):
    if not getattr(obj, method_name, None):
        setattr(obj, method_name, method_handle)
    return obj
