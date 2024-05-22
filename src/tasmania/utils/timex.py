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
from dataclasses import dataclass, field
from datetime import datetime
import functools
import numpy as np
from sympl import DataArray
import timeit
from typing import TYPE_CHECKING

from tasmania.externals import cp

if TYPE_CHECKING:
    from typing import Optional


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


@dataclass
class Node:
    label: str
    parent: Node = None
    children: dict[str, Node] = field(default_factory=dict)
    level: int = 0
    tic: float = 0
    total_calls: int = 0
    total_runtime: float = 0


class Timer:
    active: list[str] = []
    head: Optional[Node] = None
    tree: dict[str, Node] = {}

    @classmethod
    def start(cls, label: str) -> None:
        # safe-guard
        if label in cls.active:
            return

        # mark node as active
        cls.active.append(label)

        # insert timer in the tree
        node_label = cls.active[0]
        node = cls.tree.setdefault(node_label, Node(node_label))
        for i, node_label in enumerate(cls.active[1:]):
            node = node.children.setdefault(node_label, Node(node_label, parent=node, level=i + 1))
        cls.head = node

        # tic
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        cls.head.tic = timeit.default_timer()

    @classmethod
    def stop(cls, label: str = None) -> None:
        # safe-guard
        if len(cls.active) == 0:
            return

        # only nested timers allowed!
        label = label or cls.active[-1]
        assert label == cls.active[-1], f"Cannot stop {label} before stopping {cls.active[-1]}"

        # toc
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        toc = timeit.default_timer()

        # update statistics
        cls.head.total_calls += 1
        cls.head.total_runtime += toc - cls.head.tic

        # mark node as not active
        cls.active = cls.active[:-1]

        # update head
        cls.head = cls.head.parent

    @classmethod
    def reset(cls) -> None:
        cls.active = []
        cls.head = None

        def cb(node):
            node.total_calls = 0
            node.total_runtime = 0

        for root in cls.tree.values():
            cls.traverse(cb, root)

    @classmethod
    def get_time(cls, label, units="ms") -> DataArray:
        nodes = cls.get_nodes_from_label(label)
        assert len(nodes) > 0, f"{label} is not a valid timer identifier."

        raw_time = functools.reduce(lambda x, node: x + node.total_runtime, nodes, 0)
        time = DataArray(raw_time, attrs={"units": "s"}).to_units(units).data.item()

        return time

    @classmethod
    def print(cls, label, units="ms") -> None:
        time = cls.get_time(label, units)
        print(f"{label}: {time:.3f} {units}")

    @classmethod
    def log(cls, logfile: str = "log.txt", units: str = "ms") -> None:
        # ensure all timers have been stopped
        assert len(cls.active) == 0, "Some timers are still running."

        # callback
        def cb(node, out, units, prefix="", has_peers=False):
            level = node.level
            prefix_now = prefix + "|- " if level > 0 else prefix
            time = DataArray(node.total_runtime, attrs={"units": "s"}).to_units(units).data.item()
            out.write(f"{prefix_now}{node.label}: {time:.3f} {units}\n")

            prefix_new = prefix if level == 0 else prefix + "|  " if has_peers else prefix + "   "
            peers_new = len(node.children)
            has_peers_new = peers_new > 0
            for i, label in enumerate(node.children.keys()):
                cb(
                    node.children[label],
                    out,
                    units,
                    prefix=prefix_new,
                    has_peers=has_peers_new and i < peers_new - 1,
                )

        # write to file
        with open(logfile, "w") as outfile:
            for root in cls.tree.values():
                cb(root, outfile, units)

    @staticmethod
    def traverse(cb, node, **kwargs) -> None:
        cb(node, **kwargs)
        for child in node.children.values():
            Timer.traverse(cb, child, **kwargs)

    @classmethod
    def get_nodes_from_label(cls, label) -> list[Node]:
        out = []

        def cb(node, out):
            if node.label == label:
                out.append(node)

        for root in cls.tree.values():
            Timer.traverse(cb, root, out=out)

        return out
