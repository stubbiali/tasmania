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

all: clean parser html latex uml

parser:
	@cd $(PARSERDIR) && $(MAKE) 

.PHONY: clean html latex uml

clean:
	@$(RM) $(PYTMP)
	@cd $(PARSERDIR) && $(MAKE) clean

html:
	@cd $(DOCDIR) && $(MAKE) html
	@cp -r $(DOCDIR)/build/html ../meetings/20180208_phd_meeting

latex:
	@cd $(DOCDIR) && $(MAKE) latex

uml:
	@pyreverse -p grids -o eps grids/
	@pyreverse -p dycore -f OTHER -o eps dycore/
	@pyreverse -p storages -o eps storages/
	@pyreverse -p interface -o eps .
	@mv classes_*.eps $(UMLDIR)
	@mv packages_*.eps $(UMLDIR)
	#@cp $(UMLDIR)/classes_*.eps ../meetings/20180208_phd_meeting/uml
	#@cp $(UMLDIR)/packages_*.eps ../meetings/20180208_phd_meeting/uml
