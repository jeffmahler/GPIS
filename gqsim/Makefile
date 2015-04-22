.PHONY: all clean gqsim debug

all: gqsim

clean:
	rm -rf build
	if test -e filechooser.py; then rm filechooser.py; fi;
	if test -e gqsim; then rm gqsim; fi; 

gqsim: build/Makefile
	cd build;make --no-print-directory
	cp build/gqsim .
	cp 3rdparty/util/src/filechooser.py .

debug: build_debug/Makefile
	cd build_debug;make --no-print-directory
	cp build_debug/gqsim ./gqsim_d
	cp 3rdparty/util/src/filechooser.py .

build/Makefile: CMakeLists.txt
	mkdir -p build;cd build;cmake -DCMAKE_BUILD_TYPE=Release ..

build_debug/Makefile: CMakeLists.txt
	mkdir -p build_debug;cd build_debug;cmake -DCMAKE_BUILD_TYPE=Debug ..

