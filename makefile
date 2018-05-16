PYSRC := $(shell find $(shell pwd) -name '*.py')
PYTMP := $(shell find $(shell pwd) -name '*.pyc')
DOCDIR := docs
DOCSRC := $(DOCDIR)/source/conf.py $(DOCDIR)/source/api.rst
PARSERDIR := grids/parser
UMLDIR := $(DOCDIR)/uml

all: clean parser html

parser:
	@cd $(PARSERDIR) && $(MAKE)

html: parser
	@echo -n "Building HTML documentation ... "
	@cd $(DOCDIR) && $(MAKE) html > /dev/null 2>&1
	@echo "OK."

latex: parser
	@echo -n "Building LaTeX documentation ..."
	@cd $(DOCDIR) && $(MAKE) latex > /dev/null 2>&1
	@echo "OK."

latexpdf: parser
	@echo -n "Building LaTeX documentation ..."
	@cd $(DOCDIR) && $(MAKE) latexpdf > /dev/null
	@echo "OK."

uml: parser
	@echo -n "Building UML diagrams ... "
	@pyreverse -p grids -o eps grids/ > /dev/null
	@pyreverse -p dycore -f OTHER -o eps dycore/ > /dev/null
	@pyreverse -p storages -o eps storages/ > /dev/null
	@mv classes_*.eps $(UMLDIR) > /dev/null
	@mv packages_*.eps $(UMLDIR) > /dev/null
	@echo "OK."
	
.PHONY: clean 

clean:
	@$(RM) $(PYTMP) > /dev/null
	@cd $(PARSERDIR) && $(MAKE) clean > /dev/null
	@cd $(PARSERDIR)/tests && $(MAKE) clean > /dev/null

