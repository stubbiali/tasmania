PYSRC := $(shell find $(shell pwd) -name '*.py')
PYTMP_FILES := $(shell find $(shell pwd) -name '*.pyc')
PYTMP_FOLDERS := $(shell find $(shell pwd) -name '__pycache__')
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
	
.PHONY: clean gitignore

clean:
	@$(RM) $(PYTMP_FILES) > /dev/null
	@$(RM) -r $(PYTMP_FOLDERS) > /dev/null
	@find . -type f -name "*.sw[klmnop]" -delete
	@cd $(PARSERDIR) && $(MAKE) clean > /dev/null
	@cd $(PARSERDIR)/tests && $(MAKE) clean > /dev/null

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
	@find . -size +100M | cat >> .gitignore
	@echo '\n# END OF AUTOMATICALLY GENERATED TEXT'						>> .gitignore
