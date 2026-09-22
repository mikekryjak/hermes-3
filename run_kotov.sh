#!/bin/bash

lbl='low_n_srcs'
run_dir_suffix='_nod2p_long_lowndt1e5'
build='kotovLowNSrcs'
NMPI=1

build_dir="$H3/builds/$build"
data_dir="$build_dir/tests/integrated/1D-kotov/data"
exec="$build_dir/hermes-3"

mkdir -p "$ARCHH3/kotov_testing"
run_dir="$ARCHH3/kotov_testing/${lbl}$run_dir_suffix"

if [ -d "$run_dir" ]; then
    echo "Run directory $run_dir already exists, overwrite it? (y/n)"
    read answer
    if [ "$answer" != "y" ]; then
        echo "Aborted"
        exit 1
    fi
    rm -rf "$run_dir"
fi
cp -r "$data_dir" "$run_dir"

echo "    exec : $exec"
echo "    NMPI : $NMPI"
echo " run_dir : $run_dir"

cd "$run_dir"
mpirun -n $NMPI $exec -d .
