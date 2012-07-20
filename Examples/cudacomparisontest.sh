#!/bin/bash
#$Id$

# system file and simulation parameters
file="Simulators/errors_levenshtein-bsid-dminner2_precompute_modvec6_4_8_ld3-uncoded-gf16-50.txt"
par="1e-2"
seed="0"
# branch name
branch=`dirname $PWD`
branch=`basename $branch`

# Make sure all system files are built
make all

# Determine default and plain tags
plaintag=`make --no-print-directory -C .. plain-tag`
defaulttag=`make --no-print-directory -C .. tag`

for tag in "$plaintag" "$defaulttag"; do
   echo "Running test for build $tag"
   showerrorevent.$tag.debug -p $par -s $seed -i $file >$tag.log 2>&1
done
