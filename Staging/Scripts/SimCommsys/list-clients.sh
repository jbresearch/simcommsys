#!/bin/bash

ps xu |grep "simcommsys.*.release -q" |grep -v bash |tr -s ' ' '\t' |cut -f 2
