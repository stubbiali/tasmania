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
import sys
from setuptools import setup, Extension


if sys.version_info.major < 3:
    print("Python 3.x is required.")
    sys.exit(1)


setup_kwargs = {}


if not os.environ.get("DISABLE_TASMANIA_CEXT"):
    setup_kwargs["ext_package"] = "tasmania.cpp.parser"
    setup_kwargs["ext_modules"] = [
        Extension(
            "parser_1d",
            sources=[
                "tasmania/cpp/parser/parser_1d_cpp.cpp",
                "tasmania/cpp/parser/parser_1d.cpp",
            ],
            include_dirs=["tasmania/cpp/parser"],
            optional=True,
        ),
        Extension(
            "parser_2d",
            sources=[
                "tasmania/cpp/parser/parser_2d_cpp.cpp",
                "tasmania/cpp/parser/parser_2d.cpp",
            ],
            include_dirs=["tasmania/cpp/parser"],
            optional=True,
        ),
    ]


setup(use_scm_version=True, **setup_kwargs)
