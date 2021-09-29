# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import pathlib
import shutil
import tempfile


license_header_name = "LICENSE_HEADER.cpp"
single_line_header = "jefnbekwhfb"
file_pattern = "**/*.hpp"
exclude_patterns = (
    license_header_name,
    "docker/external/gridtools4py",
    "exprtk",
    "gridtools/",
    "gt_cache",
    "remove_license_header",
    "venv",
)


def with_header(source_name, header_name):
    with open(header_name, "r") as header:
        with open(source_name, "r") as source:
            for header_line in header:
                if source.readline() != header_line:
                    return False
    return True


if __name__ == "__main__":
    for pyfile_name in pathlib.Path(".").glob(file_pattern):
        pyfile_name = str(pyfile_name)

        neglect = (
            pyfile_name == os.path.basename(__file__)
            or os.path.islink(pyfile_name)
            or os.path.isdir(pyfile_name)
            or any(pattern in pyfile_name for pattern in exclude_patterns)
            or with_header(pyfile_name, license_header_name)
        )

        if not neglect:
            print("Processing {} ... ".format(pyfile_name), end="")

            with open(pyfile_name, "r") as pyfile:
                first_line = pyfile.readline()
                first_line_is_header = single_line_header in first_line

                _, new_pyfile_name = tempfile.mkstemp()
                with open(new_pyfile_name, "w") as new_pyfile:
                    with open(license_header_name, "r") as license_header:
                        for line in license_header:
                            new_pyfile.write(line)

                    if not first_line_is_header:
                        new_pyfile.write(first_line)

                    for line in pyfile:
                        new_pyfile.write(line)

            shutil.move(new_pyfile_name, pyfile_name)

            print("OK")
