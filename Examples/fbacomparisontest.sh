#!/bin/bash
#$Id$

# system file and simulation parameters
prefix="Timers/bsid-dminner2_"
suffix="modvec6_4_8_ld3-uncoded-gf16-50.txt"
par="1e-2"
time="10"

# build tag
if [[ $# < 1 ]]; then
   branch=`dirname $PWD`
   branch=`basename $branch`
   tag="$branch"
else
   tag="$1"
fi

# Make sure all system files are built
make all

# Start the tests
echo "Running tests for build $tag"

file=timings-batch-$tag.log
echo "Results for batch interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

suffix=nobatch_$suffix
file=timings-single-$tag.log
echo "Results for independent interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null
