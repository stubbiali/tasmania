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
import inspect
import math

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

try:
    from tasmania.conf import tol as d_tol
except (ImportError, ModuleNotFoundError):
    d_tol = 1e-10


def equal_to(a, b, tol=d_tol):
    """
    Compare floating point numbers, or arrays of floating point numbers,
    properly accounting for round-off errors.

    Parameters
    ----------
    a : `float` or `array_like`
        Left-hand side.
    b : `float` or `array_like`
        Right-hand side.
    tol : `float`, optional
        Tolerance.

    Return
    ------
    bool :
        ``True`` if `a` is equal to `b` up to `tol`,
        ``False`` otherwise.
    """
    return math.fabs(a - b) <= tol


def smaller_than(a, b, tol=d_tol):
    """
    Compare floating point numbers, or arrays of floating point numbers,
    properly accounting for round-off errors.

    Parameters
    ----------
    a : `float` or `array_like`
        Left-hand side.
    b : `float` or `array_like`
        Right-hand side.
    tol : `float`, optional
        Tolerance.

    Return
    ------
    bool :
        ``True`` if `a` is smaller than `b` up to `tol`,
        ``False`` otherwise.
    """
    return a < (b - tol)


def smaller_or_equal_than(a, b, tol=d_tol):
    """
    Compare floating point numbers or arrays of floating point numbers,
    properly accounting for round-off errors.

    Parameters
    ----------
    a : `float` or `array_like`
        Left-hand side.
    b : `float` or `array_like`
        Right-hand side.
    tol : `float`, optional
        Tolerance.

    Return
    ------
    bool :
        ``True`` if `a` is smaller than or equal to `b`
        up to `tol`, ``False`` otherwise.
    """
    return a <= (b + tol)


def greater_than(a, b, tol=d_tol):
    """
    Compare floating point numbers, or arrays of floating point numbers,
    properly accounting for round-off errors.

    Parameters
    ----------
    a : `float` or `array_like`
        Left-hand side.
    b : `float` or `array_like`
        Right-hand side.
    tol : `float`, optional
        Tolerance.

    Return
    ------
    bool :
        ``True`` if `a` is greater than `b` up to `tol`,
        ``False`` otherwise.
    """
    return a > (b + tol)


def greater_or_equal_than(a, b, tol=d_tol):
    """
    Compare floating point numbers, or arrays of floating point numbers,
    properly accounting for round-off errors.

    Parameters
    ----------
    a : `float` or `array_like`
        Left-hand side.
    b : `float` or `array_like`
        Right-hand side.
    tol : `float`, optional
        Tolerance.

    Return
    ------
    bool :
        ``True`` if `a` is greater than or equal to `b`
        up to `tol`, ``False`` otherwise.
    """
    return a >= (b - tol)


def assert_sequence(seq, reflen=None, reftype=None):
    """
    Assert if a sequence has appropriate length and contains objects
    of appropriate type.

    Parameters
    ----------
    seq : sequence
        The sequence.
    reflen : int
        The reference length.
    reftype : obj
        The reference type, or a list of reference types.
    """
    if reflen is not None:
        assert (
            len(seq) == reflen
        ), "The input sequence has length {}, but {} was expected.".format(
            len(seq), reflen
        )

    if reftype is not None:
        if type(reftype) is not tuple:
            reftype = (reftype,)
        for item in seq:
            error_msg = (
                "An item of the input sequence is of type "
                + str(type(item))
                + ", but one of [ "
            )
            for reftype_ in reftype:
                error_msg += str(reftype_) + " "
            error_msg += "] was expected."

            assert isinstance(item, reftype), error_msg


def feed_module(target, source, exclude_paths=None):
    def get_symbol(name, symbols):
        for symbol in symbols:
            if symbol[0] == name:
                return symbol[1]
        return None

    exclude_paths = exclude_paths or ("__",)

    source_symbols = inspect.getmembers(source)
    source_symbol_names = set(
        symbol[0]
        for symbol in source_symbols
        if all(exclude_path not in symbol[0] for exclude_path in exclude_paths)
    )
    target_symbols = inspect.getmembers(target)
    target_symbol_names = set(
        symbol[0]
        for symbol in target_symbols
        if all(exclude_path not in symbol[0] for exclude_path in exclude_paths)
    )

    missing_symbol_names = source_symbol_names.difference(target_symbol_names)

    for symbol_name in missing_symbol_names:
        symbol_value = get_symbol(symbol_name, source_symbols)
        # assert symbol_value is not None

        print(
            "Symbol '{symbol_name}' added to the module '{module_name}'.".format(
                symbol_name=symbol_name, module_name=target.__name__
            )
        )

        setattr(target, symbol_name, symbol_value)

    return target
