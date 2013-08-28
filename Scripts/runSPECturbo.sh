#!/bin/bash
# $Author$
# $Revision$
# $Date$

# Usage: runSPECturbo [Port [Count [Release]]]

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

BRANCH=`make tag`

# start slave processes (one/CPU)
declare -i i
for (( i=0; $i < $COUNT; i++ )); do
   echo Starting slave $i
   ( sleep 1; SPECturbo.$BRANCH.$RELEASE -q -p 0 -e localhost:$PORT >/dev/null 2>&1 )&
done

# start master process
echo Starting master
SPECturbo.$BRANCH.$RELEASE -e :$PORT
