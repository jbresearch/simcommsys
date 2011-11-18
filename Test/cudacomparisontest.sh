#!/bin/bash
#$Id$

# system file and simulation parameters
file="Simulators/bsid-dminner2_4_8_h-uncoded-gf16-50-float.txt"
par="1e-2"
seed="0"
# branch name
branch=`dirname $PWD`
branch=`basename $branch`

make

tag="$branch"
echo "Running test for build $tag"
showerrorevent.$tag.debug -p $par -s $seed -i $file >$tag.log 2>&1

tag="$branch-mpi-gmp-cuda11"
echo "Running test for build $tag"
showerrorevent.$tag.debug -p $par -s $seed -i $file >$tag.log 2>&1
