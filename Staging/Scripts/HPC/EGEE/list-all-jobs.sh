#!/bin/bash

edg-job-status --all --vo eumed |grep "Current Status" |tr -s ' ' '\t' |cut -f 3 |sort |uniq -c
