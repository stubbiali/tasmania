all: parser_1d parser_2d

parser_1d: parser_1d_cpp.hpp parser_1d_cpp.cpp parser_2d.pyx setup_2d.py
	python setup_1d.py build_ext --inplace	

parser_2d: parser_2d_cpp.hpp parser_2d_cpp.cpp parser_2d.pyx setup_2d.py
	python setup_2d.py build_ext --inplace	

.PHONY: clean

clean:
	$(RM) -r *.so build/
