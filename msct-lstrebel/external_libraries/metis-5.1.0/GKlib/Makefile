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
# Configuration options.
gdb      = not-set
assert   = not-set
assert2  = not-set
debug    = not-set
gprof    = not-set
openmp   = not-set
prefix   = not-set
pcre     = not-set
gkregex  = not-set
gkrand   = not-set


# Basically proxies everything to the builddir cmake.
cputype = $(shell uname -m | sed "s/\\ /_/g")
systype = $(shell uname -s)

BUILDDIR = build/$(systype)-$(cputype)

# Process configuration options.
CONFIG_FLAGS = -DCMAKE_VERBOSE_MAKEFILE=1
ifneq ($(gdb), not-set)
    CONFIG_FLAGS += -DGDB=$(gdb)
endif
ifneq ($(assert), not-set)
    CONFIG_FLAGS += -DASSERT=$(assert)
endif
ifneq ($(assert2), not-set)
    CONFIG_FLAGS += -DASSERT2=$(assert2)
endif
ifneq ($(debug), not-set)
    CONFIG_FLAGS += -DDEBUG=$(debug)
endif
ifneq ($(gprof), not-set)
    CONFIG_FLAGS += -DGPROF=$(gprof)
endif
ifneq ($(openmp), not-set)
    CONFIG_FLAGS += -DOPENMP=$(openmp)
endif
ifneq ($(pcre), not-set)
    CONFIG_FLAGS += -DPCRE=$(pcre)
endif
ifneq ($(gkregex), not-set)
    CONFIG_FLAGS += -DGKREGEX=$(pcre)
endif
ifneq ($(gkrand), not-set)
    CONFIG_FLAGS += -DGKRAND=$(pcre)
endif
ifneq ($(prefix), not-set)
    CONFIG_FLAGS += -DCMAKE_INSTALL_PREFIX=$(prefix)
endif

define run-config
mkdir -p $(BUILDDIR)
cd $(BUILDDIR) && cmake $(CURDIR) $(CONFIG_FLAGS)
endef

all clean install: $(BUILDDIR)
	make -C $(BUILDDIR) $@

uninstall:
	 xargs rm < $(BUILDDIR)/install_manifest.txt

$(BUILDDIR):
	$(run-config)

config: distclean
	$(run-config)

distclean:
	rm -rf $(BUILDDIR)

remake:
	find . -name CMakeLists.txt -exec touch {} ';'

.PHONY: config distclean all clean install uninstall remake
