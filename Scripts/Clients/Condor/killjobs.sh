#!/bin/bash

if [[ $# < 3 ]]; then
   echo "Usage: $0 <cluster> <start> <end>"
   exit
fi

declare -i i
for (( i=$2; $i <= $3; i++ )); do
   job=$1.$i
   echo Killing $job
   condor_rm $job
done
