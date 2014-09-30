#!/bin/bash

# build id
if [[ $# < 1 ]]; then
   build=`make --no-print-directory -C ../.. plain-buildid`
else
   build="$1"
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
fileprefix="qids-bool-tvb_"
filesuffix="modvec6_4_8_ld3-uncoded-gf16-50.txt"
par="1e-2"
time="10"

# Start the tests
echo "Running tests for build $build"

# Timings
prefix=Timers/$fileprefix
suffix=$filesuffix
file=timings-batch-$build-$release.log
echo "Timings for batch interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

prefix=Timers/$fileprefix
suffix=nobatch_$filesuffix
file=timings-single-$build-$release.log
echo "Timings for independent interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

# Simulations
prefix=Simulators/errors_levenshtein-random-$fileprefix
suffix=$filesuffix
file=results-batch-$build-$release.log
echo "Results for batch interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null

prefix=Simulators/errors_levenshtein-random-$fileprefix
suffix=nobatch_$filesuffix
file=results-single-$build-$release.log
echo "Results for independent interface." |tee $file
echo -e "\n\n\nPrecomputed gamma:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}precompute_${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, cached:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}${suffix} >>$file 2>/dev/null
echo -e "\n\n\nLazy gamma computation, no caching:" |tee -a $file
quicksimulation.$build.$release -t $time -r $par -i ${prefix}nocache_${suffix} >>$file 2>/dev/null
