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
