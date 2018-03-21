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
PYSRC := $(shell find $(shell pwd) -name '*.py')
PYTMP := $(shell find $(shell pwd) -name '*.pyc')
DOCDIR := docs
DOCSRC := $(DOCDIR)/source/conf.py $(DOCDIR)/source/api.rst
PARSERDIR := grids/parser
UMLDIR := $(DOCDIR)/uml

all: clean parser html latex

parser:
	@cd $(PARSERDIR) && $(MAKE)

.PHONY: clean html latex uml

clean:
	@$(RM) $(PYTMP) > /dev/null
	@cd $(PARSERDIR) && $(MAKE) clean > /dev/null
	@cd $(PARSERDIR)/tests && $(MAKE) clean > /dev/null

html:
	@echo "Building HTML documentation ..."
	@cd $(DOCDIR) && $(MAKE) html > /dev/null

latex:
	@echo "Building LaTeX documentation ..."
	@cd $(DOCDIR) && $(MAKE) latex > /dev/null

uml:
	@echo "Building UML diagrams ..."
	@pyreverse -p grids -o eps grids/ > /dev/null
	@pyreverse -p dycore -f OTHER -o eps dycore/ > /dev/null
	@pyreverse -p storages -o eps storages/ > /dev/null
	@mv classes_*.eps $(UMLDIR) > /dev/null
	@mv packages_*.eps $(UMLDIR) > /dev/null
