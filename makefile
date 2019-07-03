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
TMP_FILES := $(shell find $(shell pwd) -name '*.pyc')
TMP_FOLDERS := $(shell find $(shell pwd) -name '__pycache__') 
TMP_FOLDERS += $(shell find $(shell pwd) -name '.pytest_cache')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.idea')
TMP_FOLDERS += $(shell find $(shell pwd) -name '.cache')
BUILD_FOLDERS = build .eggs tasmania/tasmania.egg-info
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(dir $(MKFILE_PATH))
SRC_DIR := $(ROOT_DIR)/tasmania
DOC_DIR := $(ROOT_DIR)/docs
DOC_SRC := $(DOC_DIR)/source/conf.py $(DOC_DIR)/source/api.rst
PARSER_DIR := $(SRC_DIR)/cpp/parser
UML_DIR := $(DOC_DIR)/uml
TEST_DIR := $(ROOT_DIR)/tests
HYPOTHESIS_DIR := $(TEST_DIR)/.hypothesis
DOCKER_DIR := $(ROOT_DIR)/docker

.PHONY: docker-build docker-run docs uml prepare-tests tests clean distclean

docker-build:
	@cd $(DOCKER_DIR) && ./build_base.sh && echo "" && ./build_tasmania.sh

docker-run:
	@if [[ "$(shell echo $$OSTYPE)" == "linux-gnu" ]]; then\
		cd $(DOCKER_DIR) && ./run.sh;\
	elif [[ "$(shell echo $$OSTYPE)" == "darwin"* ]]; then\
		cd $(DOCKER_DIR) && ./run_mac.sh;\
	else\
		echo "Unsupported host OS.";\
	fi

docs: 
	@echo -n "Building HTML documentation ... "
	@cd $(DOC_DIR) && $(MAKE) html > /dev/null 2>&1
	@echo "OK."
	@echo -n "Building LaTeX documentation ... "
	@cd $(DOC_DIR) && $(MAKE) latex > /dev/null 2>&1
	@echo "OK."
	@echo -n "Building LaTeX-pdf documentation ... "
	@cd $(DOC_DIR) && $(MAKE) latexpdf > /dev/null
	@echo "OK."

uml:
	@echo -n "Building UML diagrams ... "
	@pyreverse -p grids -o eps grids/ > /dev/null
	@pyreverse -p dycore -f OTHER -o eps dycore/ > /dev/null
	@pyreverse -p storages -o eps storages/ > /dev/null
	@mv classes_*.eps $(UML_DIR) > /dev/null
	@mv packages_*.eps $(UML_DIR) > /dev/null
	@echo "OK."

tests:
	@cd $(TEST_DIR) && pytest --mpl --cov=$(SRC_DIR) .

prepare-tests-py35:
	@cd $(TEST_DIR) && \
	 pytest --mpl-generate-path=baseline_images/py35/test_contour 			test_contour.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_contourf 			test_contourf.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_hovmoller 		test_hovmoller.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_patches			test_patches.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_plot				test_plot.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_plot_composite	test_plot_composite.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_profile 			test_profile.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_quiver 			test_quiver.py && \
	 pytest --mpl-generate-path=baseline_images/py35/test_timeseries 		test_timeseries.py

prepare-tests-py36:
	@cd $(TEST_DIR) && \
	 pytest --mpl-generate-path=baseline_images/py36/test_contour 			test_contour.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_contourf 			test_contourf.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_hovmoller 		test_hovmoller.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_patches			test_patches.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_plot				test_plot.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_plot_composite	test_plot_composite.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_profile 			test_profile.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_quiver 			test_quiver.py && \
	 pytest --mpl-generate-path=baseline_images/py36/test_timeseries 		test_timeseries.py

prepare-tests-py37:
	@cd $(TEST_DIR) && \
	 pytest --mpl-generate-path=baseline_images/py37/test_contour 			test_contour.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_contourf 			test_contourf.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_hovmoller 		test_hovmoller.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_patches			test_patches.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_plot				test_plot.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_plot_composite	test_plot_composite.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_profile 			test_profile.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_quiver 			test_quiver.py && \
	 pytest --mpl-generate-path=baseline_images/py37/test_timeseries 		test_timeseries.py
	
clean:
	@$(RM) $(TMP_FILES) > /dev/null
	@$(RM) -r $(TMP_FOLDERS) > /dev/null
	@$(RM) -r $(HYPOTHESIS_DIR) > /dev/null
	@find . -type f -name "*.sw[klmnop]" -delete
	@$(RM) $(TEST_DIR)/.hypothesis

distclean: clean
	@cd $(PARSER_DIR) && $(MAKE) clean > /dev/null
	@cd $(PARSER_DIR)/tests && $(MAKE) clean > /dev/null
	@$(RM) -r $(BUILD_FOLDERS) > /dev/null

gitignore:
	@sed -n '/# AUTOMATICALLY GENERATED TEXT/q;p' .gitignore | cat > .gitignore_tmp
	@cp .gitignore_tmp .gitignore
	@$(RM) .gitignore_tmp
	@echo '# AUTOMATICALLY GENERATED TEXT' 					  			>> .gitignore
	@echo '# The text after this comment and up to the end of file' 	>> .gitignore
	@echo '# has been automatically generated by the gitignore target'	>> .gitignore
	@echo '# of the makefile. To prevent this target from failing,'		>> .gitignore
	@echo '# no modifications should be applied by non-expert users.\n'	>> .gitignore
	@echo '# Files which exceed maximum size allowed by GitHub'			>> .gitignore 
	@find . -size +50M | cat 											>> .gitignore
	@echo '\n# END OF AUTOMATICALLY GENERATED TEXT'						>> .gitignore
