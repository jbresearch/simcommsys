#!/bin/bash

if [[ $# < 3 ]]; then
   echo "Usage: $0 <count> <executable> \"<arguments>\""
   exit
fi

count=$1
executable=$2
arguments=$3

echo Starting $count jobs
condor_submit -a "count = $count" \
              -a "executable = $executable" \
              -a "arguments = $arguments" generic.condor
