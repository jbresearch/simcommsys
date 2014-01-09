#!/bin/bash
#
# Original authors:
#     Johann Briffa   <j.briffa@ieee.org>
#     Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
#
# Date: Apr 2007

########################
# ENVIRONMENT VARIABLES
########################
if ( which edg-job-submit > /dev/null 2>&1 ); then
	JOB_SUBMIT=edg-job-submit
elif ( which globus-job-submit > /dev/null 2>&1 ); then
	JOB_SUBMIT=globus-job-submit
else
	echo "Cannot find a job-submit program to use."
	exit 1
fi
echo "Using: $JOB_SUBMIT"
#JOB_SUBMIT_DEBUG_OPT=--debug
JDL_TEMPLATE=template_wrkpool.jdl

if [ $# -ne 3 ]; then
	echo "Usage: $0 <executable> <number of workers> <IP_or_hostname:local_port>" 1>&2
	exit 1
fi

# Command-line arguments
WRKPOOL_EXECUTABLE=$1
WRKPOOL_WORKER_CNT=$2
WRKPOOL_ARGS="-q -p 0 $3"

WRKPOOL_COMMAND=$(basename $WRKPOOL_EXECUTABLE)
WRKPOOL_EXECUTABLE=$(echo $WRKPOOL_EXECUTABLE|sed 's/\//\\\//g')

echo WRKPOOL_COMMAND: $WRKPOOL_COMMAND
# Create JDL from template
JDL_FILE=wrkpool_$$.jdl

sed "s/WRKPOOL_EXECUTABLE/$WRKPOOL_EXECUTABLE/g" < $JDL_TEMPLATE|
	sed "s/WRKPOOL_COMMAND/$WRKPOOL_COMMAND/g"|
	sed "s/WRKPOOL_OUTPUT_FILE/$WRKPOOL_COMMAND.out/g"|
	sed "s/WRKPOOL_ERROR_FILE/$WRKPOOL_COMMAND.err/g"|
	sed "s/WRKPOOL_ARGS/$WRKPOOL_ARGS/g" >$JDL_FILE

# File where Grid Job IDs are kept for future reference
JOBIDS_FILE=wrkpool_$$.jid

# Submit it
for i in $(seq 1 1 $WRKPOOL_WORKER_CNT); do
	echo '----->' Ready to submit job $i of $WRKPOOL_WORKER_CNT 1>&2
	$JOB_SUBMIT $JOB_SUBMIT_DEBUG_OPT -o $JOBIDS_FILE $JDL_FILE
	if [ $? -ne 0 ]; then
		echo Job submission failed, bailing out... 1>&2
		exit 1
	fi
done

# Remove JDL file only if jobs submitted ok
rm -f $JDL_FILE

