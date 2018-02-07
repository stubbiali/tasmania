all: test1d

test1d: ../parser_1d_cpp.hpp ../parser_1d_cpp.cpp test_parser_1d_cpp.cpp
	g++ -std=c++11 -I ../ -c -o parser_1d_cpp.o ../parser_1d_cpp.cpp
	g++ -std=c++11 -I ../ -c -o test_parser_1d_cpp.o test_parser_1d_cpp.cpp
	g++ -std=c++11 -I ../ -o test_parser_1d_cpp parser_1d_cpp.o test_parser_1d_cpp.o

.PHONY: clean

clean:
	$(RM) *.o test_parser_1d
