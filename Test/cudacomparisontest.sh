#!/bin/bash
#$Id$

make
file="Simulators/bsid-dminner2_4_8_h-uncoded-gf16-50-float.txt"
tag=`dirname $PWD`
tag=`basename $tag`
echo "Running test for build $tag"
showerrorevent.$tag.debug -p 1e-2 -s 0 -i $file >$tag.log 2>&1
tag="$tag-mpi-gmp-cuda11"
echo "Running test for build $tag"
showerrorevent.$tag.debug -p 1e-2 -s 0 -i $file >$tag.log 2>&1
