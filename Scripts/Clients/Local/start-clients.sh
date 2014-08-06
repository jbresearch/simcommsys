#!/bin/bash

if (( $# < 4 )); then
   echo "Usage: ${0##*/} <tag> <count> <host> <first port> [<last port>]"
   echo "<tag> is the build tag name (XX in simcommsys.XX.release)"
   echo "<count> is the number of clients to start per server"
   echo "<host> is the hostname on which the server is running"
   echo "<first port> is the first port to use (inclusive)"
   echo "<last port> is the last port to use (inclusive)"
   echo "Simulations are run in screen processes named <host.port.client>"
   echo
   echo "Called with $# parameters:"
   while (( $# > 0 )); do
      echo "  Parameter:  $1"
      shift
   done
   exit
fi

# set up user arguments
tag=$1
count=$2
host=$3
first=$4
if [[ $# < 5 ]]; then last=$4; else last=$5; fi
if [[ $last < $first ]]; then t=$last; last=$first; first=$t; fi
# determine release suffix to use
if [ -e "$(which simcommsys.$tag.release)" ]; then
   release="release"
elif [ -e "$(which simcommsys.$tag.Release)" ]; then
   release="Release"
else
   echo "Cannot find release binary with tag $tag."
   exit 1
fi
# determine full path to submitter script
submit="${0%/*}/submit-local.sh"

# start the clients
declare -i port
for (( port=$first; $port <= $last; port++ )); do
   echo Starting $count jobs on $host:$port
   declare -i client
   for (( client=0; $client < $count; client++ )); do
      screen -d -m -S "$host.$port.$client" "$submit" "simcommsys.$tag.$release" -q -p 0 -e "$host:$port"
   done
done
