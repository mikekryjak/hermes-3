#!/bin/bash

lbl='NPD_all'
build='kotov'
NMPI=5

build_dir="$H3/builds/$build"
data_dir="$H3/builds/$build/tests/integrated/1D-kotov/data"
exec="$build_dir/hermes-3"

mkdir -p "$ARCHH3/kotov_testing"
run_dir="$ARCHH3/kotov_testing/$lbl"

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
mpirun -n $NMPI $exec
