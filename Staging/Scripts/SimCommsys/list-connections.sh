#!/bin/bash
# $Id$

netstat -nt |tr -s ' ' '\t' |cut -f 4 |grep '\:99' |sort |uniq -c
