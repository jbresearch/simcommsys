#!/bin/bash

if [[ $# < 4 ]]; then
   echo "Usage: ${0##*/} <tag> <count> <host> <first port> [<last port>]"
   echo "<tag> is the build tag name (XX in simcommsys.XX.release)"
   echo "<count> is the number of clients to start per server"
   echo "<host> is the hostname on which the server is running"
   echo "<first port> is the first port to use (inclusive)"
   echo "<last port> is the last port to use (inclusive)"
   exit
fi

jdl=${0##*/submit-}
jdl=${jdl%.sh}
jdl=$jdl.condor

# set up user arguments
tag=$1
count=$2
host=$3
first=$4
if [[ $# < 5 ]]; then last=$4; else last=$5; fi
if [[ $last < $first ]]; then t=$last; last=$first; first=$t; fi

# start the clients
declare -i port
for (( port=$first; $port <= $last; port++ )); do
   echo Starting $count jobs on $host:$port
   echo Using $jdl
   condor_submit -a "tag = $tag" \
                 -a "count = $count" \
                 -a "host = $host" \
                 -a "port = $port" $jdl
done
