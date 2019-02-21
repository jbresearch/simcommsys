#!/bin/bash

if (( $# < 4 )); then
   echo "Usage: ${0##*/} <tag> <port> <vlimit> <other> [<systems>]"
   echo "<tag> is the build tag name (XX in simcommsys.XX.release)"
   echo "<port> is the first port to use, then decrement (or 'local')"
   echo "<vlimit> is the virtual memory limit in KiB (or 'default' or 'unlimited')"
   echo "   default = divide available memory by number of (virtual) CPUs"
   echo "<other> contains any other parameters to pass to simcommsys"
   echo "        (may be blank but must be present)"
   echo "<systems> is a list of systems to simulate in parallel"
   echo "Simulations are run in screen processes named <port.system>"
   echo
   echo "Called with $# parameters:"
   while (( $# > 0 )); do
      echo "  Parameter:  $1"
      shift
   done
   exit
fi

# set up user arguments
path=${0%/*}
tag=$1
port=$2
vlimit=$3
other=$4
shift 5
# interpret port numbering
if [[ $port != "local" ]]; then declare -i port; fi

while (( $# > 0 )); do
   systempath=$1
   system=${1##*/}
   echo "Starting simulation for:"
   echo "  System: $system"
   echo "  Tag:    $tag"
   echo "  Port:   $port"
   echo "  Args:   $other"

   screen -d -m -S "$port.$system" "$path/simcommsys-wrapper.sh" "$systempath" "$tag" "$port" "$vlimit" $other
   # interpret port numbering
   if [[ $port != "local" ]]; then (( port -- )); fi
   shift
done
