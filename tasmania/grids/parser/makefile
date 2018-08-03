all: parser_1d parser_2d

parser_1d: parser_1d_cpp.hpp parser_1d_cpp.cpp parser_2d.pyx setup_2d.py
	@echo "Compiling module parser_1d ... "
	@python3.5 setup_1d.py build_ext --inplace	> /dev/null
	@echo "Module parser_1d successfully compiled."

parser_2d: parser_2d_cpp.hpp parser_2d_cpp.cpp parser_2d.pyx setup_2d.py
	@echo "Compiling module parser_2d ..."
	@python3.5 setup_2d.py build_ext --inplace	> /dev/null
	@echo "Module parser_2d successfully compiled."

.PHONY: clean

clean:
	@$(RM) -r *.so build/ > /dev/null
