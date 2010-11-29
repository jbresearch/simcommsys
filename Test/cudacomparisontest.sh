#!/bin/bash
#$Id$

make
file="Simulators/bsid-dminner2_4_8_h-uncoded-gf16-50-float.txt"
tag="jab"
echo "Running test for build $tag"
showerrorevent.$tag.Debug -p 1e-2 -s 0 -i $file >$tag.log 2>&1
tag="jab-mpi-cuda"
echo "Running test for build $tag"
showerrorevent.$tag.Debug -p 1e-2 -s 0 -i $file >$tag.log 2>&1
