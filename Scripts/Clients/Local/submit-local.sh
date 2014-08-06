#!/bin/bash -l

if [[ $# < 1 ]]; then
   echo "Usage: $0 <executable> [<argument>..]"
   exit
fi

# set memory limit
if [ -f /proc/meminfo -a -f /proc/cpuinfo -a -e $(which gawk) ]; then
   CPUS=$(gawk '/processor/ { n++ } END { print n }' /proc/cpuinfo)
   vlimit=$(gawk -v n="$CPUS" '/MemTotal/ { printf "%d",$2/n }' /proc/meminfo)
   echo "Setting memory limit to $vlimit KiB"
   ulimit -v $vlimit
fi

# determine main parameters
executable=`which $1`
shift 1
# determine command arguments
arguments=""
for arg in "$@"; do
   arguments="$arguments \"$arg\""
done

# run the command locally, redirecting I/O
$executable $@ < /dev/null > $HOME/submit-local-$$.out 2> $HOME/submit-local-$$.err

# return with job exit status
exit $?
