#!/bin/bash

if [[ $# < 1 ]]; then
   echo "Usage: ${0##*/} <port>"
   exit 1
fi

port=$1

ssh -R \*:$port:localhost:$port jabriffa@guinevere.eng.um.edu.mt
