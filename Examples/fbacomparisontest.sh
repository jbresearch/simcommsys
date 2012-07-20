#!/bin/bash
#$Id$

# build tag
if [[ $# < 1 ]]; then
   tag=`make --no-print-directory -C .. plain-tag`
else
   tag="$1"
fi

# build release
if [[ $# < 2 ]]; then
   release="release"
else
   release="$2"
fi

# Make sure all system files are built
make all

# system file and simulation parameters
fileprefix="bsid-dminner2_"
filesuffix="modvec6_4_8_ld3-uncoded-gf16-50.txt"
par="1e-2"
time="10"

# Start the tests
echo "Running tests for build $tag"

# Timings
prefix=Timers/$fileprefix
suffix=$filesuffix
file=timings-batch-$tag-$release.log
echo "Timings for batch interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

prefix=Timers/$fileprefix
suffix=nobatch_$filesuffix
file=timings-single-$tag-$release.log
echo "Timings for independent interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

# Simulations
prefix=Simulators/errors_levenshtein-$fileprefix
suffix=$filesuffix
file=results-batch-$tag-$release.log
echo "Results for batch interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

prefix=Simulators/errors_levenshtein-$fileprefix
suffix=nobatch_$filesuffix
file=results-single-$tag-$release.log
echo "Results for independent interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
SPECturbo.$tag.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null
