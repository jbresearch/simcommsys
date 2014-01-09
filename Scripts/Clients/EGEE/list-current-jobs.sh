#!/bin/bash

for file in wrkpool_*.jid; do
   echo In ${file}:
   echo |edg-job-status -i $file |grep "Current Status" |tr -s ' ' '\t' |cut -f 3 |sort |uniq -c
done
