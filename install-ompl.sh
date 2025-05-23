#!/bin/bash
set -e

if [ `id -u` == 0 ]; then
    export SUDO=
    export DEBIAN_FRONTEND=noninteractive
else
    SUDO="sudo -H"
fi

# Get the conda environment information
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active conda environment detected."
    echo "Please activate your conda environment with 'conda activate your_env_name' first."
    exit 1
fi

echo "Using conda environment: $CONDA_PREFIX"

install_common_dependencies() {
    # install most dependencies via apt-get
    ${SUDO} apt-get -y update && \
    ${SUDO} apt-get -y install \
        clang \
        cmake \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libexpat1 \
        libtriangle-dev \
        ninja-build \
        pkg-config \
        wget
    export CXX=clang++
}

install_conda_python_dependencies() {
    # Install Python dependencies via conda and pip in the conda environment
    conda install -y -c conda-forge \
        boost-cpp \
        numpy \
        eigen \
        pyqt \
        matplotlib \
        boost \
        pybind11

    # Install castxml
    ${SUDO} apt-get -y install castxml

    # Install pygccxml and pyplusplus in the conda environment
    pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus
}

install_app_dependencies() {
    ${SUDO} apt-get -y install \
        freeglut3-dev \
        libassimp-dev \
        libccd-dev \
        libfcl-dev
}

install_ompl() {
    if [ -z $APP ]; then
        wget -O - https://github.com/ompl/ompl/archive/1.7.0.tar.gz | tar zxf -
        cd ompl-1.7.0
    else
        wget -O - https://github.com/ompl/omplapp/releases/download/1.7.0/omplapp-1.7.0-Source.tar.gz | tar zxf -
        cd omplapp-1.7.0-Source
    fi
    
    # Use conda environment's Python
    PYTHON_EXEC=$CONDA_PREFIX/bin/python
    PYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").so
    
    cmake \
        -G Ninja \
        -B build \
        -DPYTHON_EXEC=$PYTHON_EXEC \
        -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
        -DOMPL_REGISTRATION=OFF \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    cmake --build build -t update_bindings && \
    cmake --build build && \
    cmake --install build
}

for i in "$@" 
do 
case $i in
    -a|--app)
        APP=1
        PYTHON=1
        shift
        ;;
    -p|--python)
        PYTHON=1
        shift
        ;;
    *)
        # unknown option -> show help
        echo "Usage: `basename $0` [-p] [-a]"
        echo "  -p: enable Python bindings"
        echo "  -a: enable OMPL.app (implies '-p')"
    ;;
esac
done

install_common_dependencies
if [ ! -z $PYTHON ]; then
    install_conda_python_dependencies
fi
if [ ! -z $APP ]; then
    install_app_dependencies
fi
install_ompl