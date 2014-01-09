#!/bin/bash
#
# Original authors:
#     Johann Briffa   <j.briffa@ieee.org>
#     Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
#
# Date: Apr 2007

if ( which edg-job-get-output > /dev/null 2>&1 ); then
        JOB_GET_OUTPUT=edg-job-get-output
elif ( which globus-job-get-output > /dev/null 2>&1 ); then
        JOB_GET_OUTPUT=globus-job-get-output
else
        echo "Cannot find a job-get-output program to use."
        exit 1
fi
echo "Using: $JOB_GET_OUTPUT"

for file in $( find . -name "wrkpool_*.jid" ); do
	jid=${file%.jid}
	jid=output_${jid#*wrkpool_}
	echo '----->' Ready to get output from job list in $file 1>&2
	mkdir -p $jid
	echo |$JOB_GET_OUTPUT --dir $jid -i $file
	if [ $? -ne 0 ]; then
		echo Job get output failed, bailing out... 1>&2
		continue
	fi
	rm -f $file
done
