#!/bin/bash

# Usage: benchmark.sh [Port [Count [Release]]]

if (( $# > 0 )); then
   PORT=$1
else
   PORT=9999
fi

if (( $# > 1 )); then
   COUNT=$2
else
   COUNT=`grep processor /proc/cpuinfo |wc -l`
fi

if (( $# > 2 )); then
   RELEASE=$3
else
   RELEASE=release
fi

BUILD=`make --no-print-directory -C .. buildid`

# start slave processes (one/CPU)
declare -i i
for (( i=0; $i < $COUNT; i++ )); do
   echo Starting slave $i
   ( sleep 1; quicksimulation.$BUILD.$RELEASE -q -p 0 -e localhost:$PORT >/dev/null 2>&1 )&
done

# start master process
echo Starting master
quicksimulation.$BUILD.$RELEASE -e :$PORT -r 0.5 -i Simulators/errors_hamming-random-awgn-bpsk-turbo-rscc_1_2_2-helical-158-2-stipple-10.txt
