#!/bin/bash -l

if (( $# < 3 )); then
   echo "Usage: ${0##*/} [<path to>/]<system> <tag> <port> [<vlimit>] [<other>]"
   echo "Input is taken from Simulators/<system>"
   echo "Output is appended to Results/<sim>.<tag>.<system>"
   echo "<port> is the port to attach the server to (or 'local')"
   echo "<vlimit> is the virtual memory limit in KiB (or 'default' or 'unlimited')"
   echo "   default = divide available memory by number of (virtual) CPUs"
   echo "<other> contains any other parameters to pass to simcommsys"
   echo "Defaults: use simcommsys defaults)"
   exit
fi

# interpret user parameters
system=$1
tag=$2
port=$3
shift 3
if (( $# > 0 )); then
   vlimit=$1
   shift
else
   vlimit="default"
fi
# interpret port numbering
if [[ $port != "local" ]]; then port=:$port; fi
# interpret simulation type
sim=$(basename ${system%/*})
sim=${sim:0:3} # first three characters
sim=$(echo $sim |tr '[A-Z]' '[a-z]') # convert to lowercase
if [ -e "$sim" ]; then sim=sim; fi
# set filenames
infile=$system
outfile=Results/$sim.$tag.${system##*/}
# determine release suffix to use
if [ -e "$(which simcommsys.$tag.release)" ]; then
   release="release"
elif [ -e "$(which simcommsys.$tag.Release)" ]; then
   release="Release"
else
   echo "Cannot find release binary with tag $tag."
   exit 1
fi

# set memory limit
if [ "$vlimit" == "default" ]; then
   if [ -f /proc/meminfo -a -f /proc/cpuinfo ]; then
      CPUS=$(gawk '/processor/ { n++ } END { print n }' /proc/cpuinfo)
      vlimit=$(gawk -v n="$CPUS" '/MemTotal/ { printf "%d",$2/n }' /proc/meminfo)
      echo "Setting memory limit to $vlimit KiB"
      ulimit -v $vlimit
   fi
else
   echo "Setting memory limit to $vlimit KiB"
   ulimit -v $vlimit
fi

# run the actual simulation
echo "Starting simulation for:"
echo "  System: $system"
echo "  Tag:    $tag"
echo "  Port:   $port"
echo "  Args:   $@"

simcommsys.$tag.$release -i "$infile" -o "$outfile" -e $port $@

# check for errors
if (( $? > 0 )); then read -p "Press return to exit."; fi
