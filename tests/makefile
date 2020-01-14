# -*- Makefile -*-
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
TMP_FILES := $(shell find $(shell pwd) -name '*.pyc')
TMP_FILES += $(shell find $(shell pwd) -name '.coverage')
TMP_FOLDERS := $(shell find $(shell pwd) -name '__pycache__')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.pytest_cache')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.idea')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.cache')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.hypothesis')
GTCACHE := $(shell find $(shell pwd) -name '.gt_cache')
MPL_FOLDERS := baseline_images result_images
ENABLE_MPL_TESTS := 0

all: clean prepare run

clean:
	@$(RM) $(TMP_FILES)
	@$(RM) -r $(TMP_FOLDERS)

prepare:
ifeq ($(ENABLE_MPL_TESTS), 1)
	@./prepare_matplotlib.sh
endif

run: prepare
ifeq ($(ENABLE_MPL_TESTS), 1)
	@pytest --mpl --cov=../tasmania/python \
        burgers/ dwarfs/ framework/ gt4py/ isentropic/ physics/ plot/ utils/
else
	@pytest --mpl --cov=../tasmania/python \
        burgers/ dwarfs/ framework/ gt4py/ isentropic/ physics/ utils/
endif

gtclean: clean
	@$(RM) -r $(GTCACHE)

distclean: gtclean
	@$(RM) -rf $(MPL_FOLDERS)
