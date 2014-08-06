#!/bin/bash -l
#$Id: submit-local.sh 8815 2013-03-03 09:59:19Z jabriffa $

if [[ $# < 1 ]]; then
   echo "Usage: $0 <executable> [<argument>..]"
   exit
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
