#!/bin/bash

# system file and simulation parameters
file="Simulators/errors_levenshtein-random-qids-bool-tvb_precompute_modvec6_4_8_ld3-uncoded-gf16-50.txt"
par="1e-2"
seed="0"

# Make sure all system files are built
make all

# Determine default and plain builds
plainbuild=`make --no-print-directory -C ../.. plain-buildid`
defaultbuild=`make --no-print-directory -C ../.. buildid`

for build in "$plainbuild" "$defaultbuild"; do
   echo "Running test for build $build"
   showerrorevent.$build.debug -p $par -s $seed -i $file >$build.log 2>&1
done
