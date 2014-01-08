#!/bin/bash
# $Id$

qstat -qs E |tr -s ' ' '\t' |cut -f 2
