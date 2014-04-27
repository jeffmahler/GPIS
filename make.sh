#!/bin/bash

# defaults
CMAKE_BUILD_TYPE=Release
GPIS_BUILD_DIRECTORY="build"

if [[ "$(uname)" == "Linux" || "$(uname)" == "Darwin" ]]; then
    CMAKE_GENERATOR="Unix Makefiles"
else
    CMAKE_GENERATOR="MSYS Makefiles"
fi

function make_clean {
    rm -rf $GPIS_BUILD_DIRECTORY
    rm -rf bin lib update
}

while (($#)); do
    case $1 in
        -r | --release | release)
            CMAKE_BUILD_TYPE=Release
            ;;
        -d | --debug | debug)
            CMAKE_BUILD_TYPE=Debug
            ;;
        --clean | clean)
            make_clean
            exit 0
            ;;
        *)
            echo "Incorrect argument given."
            print_help
            exit 0
            ;;
    esac
    shift
done

mkdir -p $GPIS_BUILD_DIRECTORY
cd $GPIS_BUILD_DIRECTORY
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -G "$CMAKE_GENERATOR"

make -j3
cd ..
