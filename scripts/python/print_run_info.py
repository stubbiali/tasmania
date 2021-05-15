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
import functools
import os
import pandas as pd
from typing import Callable, Dict, List


backend_alias = {
    # "numpy": "NumPy",
    "gt4py:gtx86": "GT4Py x86",
    "gt4py:gtmc": "GT4Py MC",
    # "cupy": "CuPy",
    "gt4py:gtcuda": "GT4Py GPU",
}


def p_get_total_time(
    root: str, model: str, method: str, backend: str
) -> List[float]:
    filename = os.path.join(root, model + "_run_" + method + ".csv")
    df = pd.read_csv(filename, sep=",")
    return df[backend].dropna().tolist()


def p_get_gt4py_cpp_time(
    root: str, model: str, method: str, backend: str
) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_run_cpp_time"].sum()


def p_get_gt4py_run_time(
    root: str, model: str, method: str, backend: str
) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_run_time"].sum()


def p_get_gt4py_call_time(
    root: str, model: str, method: str, backend: str
) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_call_time"].sum()


def p_get_tasmania_time(
    root: str, model: str, method: str, backend: str
) -> float:
    total_time = p_get_total_time(root, model, method, backend)[-1]
    gt4py_call_time = p_get_gt4py_call_time(root, model, method, backend)
    return total_time - gt4py_call_time


def dispatcher(
    root: str, model: str, method: str, cb: Callable
) -> Dict[str, float]:
    return {
        backend: cb(root, model, method, backend) for backend in backend_alias
    }


get_gt4py_cpp_time = functools.partial(dispatcher, cb=p_get_gt4py_cpp_time)
get_gt4py_call_time = functools.partial(dispatcher, cb=p_get_gt4py_run_time)
get_gt4py_run_time = functools.partial(dispatcher, cb=p_get_gt4py_call_time)
get_tasmania_time = functools.partial(dispatcher, cb=p_get_tasmania_time)


def main(project_root, data_dir, model, method):
    root = os.path.join(project_root, data_dir)
    for backend in backend_alias:
        args = (root, model, method, backend)
        gt4py_cpp = p_get_gt4py_cpp_time(*args)
        gt4py_run = p_get_gt4py_run_time(*args)
        gt4py_call = p_get_gt4py_call_time(*args)
        taz = p_get_tasmania_time(*args)
        total = gt4py_call + taz
        print(
            f"{backend_alias[backend]:10s}: "
            f"GT4Py C++: {gt4py_cpp:.5f} ({gt4py_cpp / total * 100.0:.2f}%), "
            f"GT4Py run: {gt4py_run:.5f} ({gt4py_run / total * 100.0:.2f}%), "
            f"GT4Py call: {gt4py_call:.5f} ({gt4py_call / total * 100.0:.2f}%), "
            f"Tasmania: {taz:.5f} ({taz / total * 100.0:.2f}%)"
        )


if __name__ == "__main__":
    project_root = "/Users/subbiali/Desktop/phd/tasmania/oop"
    data_dir = "drivers/benchmarking/timing/oop"
    main(project_root, data_dir, "isentropic_moist", "fc")
