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

latex:
	@cd $(DOCDIR) && $(MAKE) latex

uml:
	@pyreverse -p grids -o eps grids/
	@pyreverse -p dycore -o eps dycore/
	@pyreverse -p storages -o eps storages/
	@pyreverse -p interface -o eps .
	@mv classes_*.eps $(UMLDIR)
	@mv packages_*.eps $(UMLDIR)
