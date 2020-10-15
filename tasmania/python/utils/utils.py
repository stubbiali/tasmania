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
from copy import deepcopy
from datetime import datetime
import inspect
import math
import numpy as np
from sympl import DataArray
import timeit
from typing import Any, Dict, List, Union

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


def convert_datetime64_to_datetime(time):
    """
    Convert :class:`numpy.datetime64` to :class:`datetime.datetime`.

    Parameters
    ----------
    time : obj
        The :class:`numpy.datetime64` object to convert.

    Return
    ------
    obj :
        The converted :class:`datetime.datetime` object.

    References
    ----------
    https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64.
    https://github.com/bokeh/bokeh/pull/6192/commits/48aea137edbabe731fb9a9c160ff4ab2b463e036.
    """
    # safeguard check
    if type(time) == datetime:
        return time

    ts = (time - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(ts)


def get_time_string(seconds, print_milliseconds=False):
    """
    Convert seconds into a string of the form hours:minutes:seconds[.milliseconds].

    Parameters
    ----------
    seconds : float
        Total seconds.
    """
    s = ""

    hours = int(seconds / (60 * 60))
    s += "{:02d}:".format(hours)
    remainder = seconds - hours * 60 * 60

    minutes = int(remainder / 60)
    s += "{:02d}:".format(minutes)
    remainder -= minutes * 60

    s += "{:02d}".format(int(remainder))

    if print_milliseconds:
        milliseconds = int(1000 * (remainder - int(remainder)))
        s += ".{:03d}".format(milliseconds)

    return s


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


def thomas_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    out: np.ndarray,
    *,
    i: Union[int, slice],
    j: Union[int, slice],
    kstart: int,
    kstop: int
):
    """ The Thomas' algorithm to solve a tridiagonal system of equations. """
    beta = deepcopy(b)
    delta = deepcopy(d)
    for k in range(kstart + 1, kstop):
        w = np.where(
            beta[i, j, k - 1] != 0.0,
            a[i, j, k] / beta[i, j, k - 1],
            a[i, j, k],
        )
        beta[i, j, k] -= w * c[i, j, k - 1]
        delta[i, j, k] -= w * delta[i, j, k - 1]

    out[i, j, kstop - 1] = np.where(
        beta[i, j, kstop - 1] != 0.0,
        delta[i, j, kstop - 1] / beta[i, j, kstop - 1],
        delta[i, j, kstop - 1] / b[i, j, kstop - 1],
    )
    for k in range(kstop - 2, kstart - 1, -1):
        out[i, j, k] = np.where(
            beta[i, j, k] != 0.0,
            (delta[i, j, k] - c[i, j, k] * out[i, j, k + 1]) / beta[i, j, k],
            (delta[i, j, k] - c[i, j, k] * out[i, j, k + 1]) / b[i, j, k],
        )


def is_gt(backend: str):
    return len(backend) > 6 and backend[:6] == "gt4py:"


def get_gt_backend(backend):
    assert is_gt(backend), f"{backend} is not a GT4Py backend."
    return backend[6:]


class Timer:
    active: List[str] = []
    roots: List[str] = []
    tic: Dict[str, Any] = {}
    tree: Dict[str, Any] = {}

    @classmethod
    def start(cls, label: str) -> None:
        # safe-guard
        assert (
            label not in cls.active
        ), f"Timer {label} has already been started."

        # mark timer as active
        cls.active.append(label)

        # insert timer in the tree
        if label not in cls.tree:
            level = len(cls.active) - 1
            parent = None if len(cls.active) == 1 else cls.active[-2]
            cls.tree[label] = {
                "level": level,
                "parent": parent,
                "children": [],
                "total_calls": 0,
                "total_runtime": 0.0,
            }
            if level == 0:
                cls.roots.append(label)
            else:
                cls.tree[parent]["children"].append(label)

        # tic
        if cp is not None:
            cp.cuda.Device(0).synchronize()
        cls.tic[label] = timeit.default_timer()

    @classmethod
    def stop(cls, label: str = None) -> None:
        # safe-guard
        if len(cls.active) == 0:
            return

        # only nested timers allowed!
        label = label or cls.active[-1]
        assert (
            label == cls.active[-1]
        ), f"Cannot stop {label} before stopping {cls.active[-1]}"

        # toc
        if cp is not None:
            cp.cuda.Device(0).synchronize()
        toc = timeit.default_timer()

        # update runtime
        cls.tree[label]["total_calls"] += 1
        cls.tree[label]["total_runtime"] += toc - cls.tic[label]

        # mark timer as not active
        cls.active = cls.active[:-1]

    @classmethod
    def reset(cls) -> None:
        cls.active = []
        cls.tic = {}
        for label in cls.tree:
            cls.tree[label]["total_calls"] = 0
            cls.tree[label]["total_runtime"] = 0.0

    @classmethod
    def print(cls, label, units="ms") -> None:
        assert label in cls.tree, f"{label} is not a valid timer identifier."
        time = (
            DataArray(cls.tree[label]["total_runtime"], attrs={"units": "s"})
            .to_units(units)
            .data.item()
        )
        print(f"{label}: {time:.3f} {units}")

    @classmethod
    def log(cls, logfile: str = "log.txt", units: str = "ms") -> None:
        # ensure all timers have been stopped
        assert len(cls.active) == 0, "Some timers are still running."

        # write to file
        with open(logfile, "w") as outfile:
            for root in cls.roots:
                cls._traverse_and_print(outfile, root, units)

    @classmethod
    def _traverse_and_print(
        cls,
        outfile,
        label: str,
        units: str,
        prefix: str = "",
        has_peers: bool = False,
    ) -> None:
        level = cls.tree[label]["level"]
        prefix_now = prefix + "|- " if level > 0 else prefix
        time = (
            DataArray(cls.tree[label]["total_runtime"], attrs={"units": "s"})
            .to_units(units)
            .data.item()
        )
        outfile.write(f"{prefix_now}{label}: {time:.3f} {units}\n")

        prefix_new = (
            prefix
            if level == 0
            else prefix + "|  "
            if has_peers
            else prefix + "   "
        )
        peers_new = len(cls.tree[label]["children"])
        has_peers_new = peers_new > 0
        for i, child in enumerate(cls.tree[label]["children"]):
            cls._traverse_and_print(
                outfile,
                child,
                units,
                prefix=prefix_new,
                has_peers=has_peers_new and i < peers_new - 1,
            )
