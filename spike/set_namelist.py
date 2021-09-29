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
import shutil


def set_namelist(user_namelist=None):
    """
    Place the user-defined namelist module in the Python search path.
    This is achieved by physically copying the content of the user-provided module into TASMANIA_ROOT/conf.py.

    Parameters
    ----------
    user_namelist : str
            Path to the user-defined namelist. If not specified, the default namelist TASMANIA_ROOT/_namelist.py is used.
    """
    try:
        tasmania_root = os.environ["TASMANIA_ROOT"]
    except RuntimeError:
        print("Hint: has the environmental variable TASMANIA_ROOT been set?")
        raise

    if user_namelist is None:  # Default case
        src_file = os.path.join(tasmania_root, "_namelist.py")
        dst_file = os.path.join(tasmania_root, "conf.py")
        shutil.copy(src_file, dst_file)
    else:
        src_dir = os.curdir
        src_file = os.path.join(src_dir, user_namelist)
        dst_file = os.path.join(tasmania_root, "conf.py")
        shutil.copy(src_file, dst_file)
