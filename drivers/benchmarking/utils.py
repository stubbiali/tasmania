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
import os
import pandas as pd
from pydantic import BaseModel

from tasmania.python.utils.backend import is_gt, get_gt_backend


class Statistics(BaseModel):
    ncalls: int = 0
    total_call_time: float = 0.0
    total_run_time: float = 0.0
    total_run_cpp_time: float = 0.0


def inject_backend(namelist, backend=None):
    if (
        backend is not None
        and getattr(namelist, "backend", backend) != backend
    ):
        old_backend = getattr(namelist, "backend")
        for key in namelist.__dict__:
            if isinstance(namelist.__dict__[key], str):
                namelist.__dict__[key] = namelist.__dict__[key].replace(
                    old_backend, backend
                )


def exec_info_to_csv(filename, backend, backend_options):
    if (
        filename is not None
        and is_gt(backend)
        and backend_options.exec_info.get("__aggregate_data", False)
    ):
        df = pd.DataFrame()

        dct = backend_options.exec_info
        gt_backend = get_gt_backend(backend)
        rows = [key for key in dct if gt_backend in key]
        cols = [
            "ncalls",
            "total_call_time",
            "total_run_time",
            "total_run_cpp_time",
        ]

        for row in rows:
            raw_stats = {col: dct[row].get(col, 0) for col in cols}
            stats = Statistics(**raw_stats)
            for col in cols:
                df.at[row, col] = getattr(stats, col)

        df.to_csv(filename)


def run_info_to_csv(filename, backend, value):
    if filename is not None:
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=",")
        else:
            df = pd.DataFrame()

        if backend not in df:
            df.at[0, backend] = value
        else:
            df.at[len(df.loc[:, backend].dropna()), backend] = value

        df.to_csv(filename, index=False)
