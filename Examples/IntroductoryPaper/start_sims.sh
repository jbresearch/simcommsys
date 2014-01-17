#!/bin/bash

settings="--start 1.5 --stop 11 --step 0.1 --floor-min 1e-5"
settings="$settings --confidence 0.95 --relative-error 0.05"

#simcommsys-batch.sh jab local "$settings" Simulators/*

#Simulators/errors_hamming-random-awgn-bpsk-concantenated-reedsolomon_255_223_gf256-map_interleaved-nrcc_133_171.txt
#Simulators/errors_hamming-random-awgn-bpsk-nrcc_133_171.txt
#Simulators/errors_hamming-random-awgn-bpsk-uncoded.txt

simcommsys-batch.sh jab 9999 "$settings" \
   Simulators/errors_hamming-random-awgn-bpsk-concantenated-reedsolomon_255_223_gf256-map_interleaved-nrcc_133_171.txt \
   Simulators/errors_hamming-random-awgn-bpsk-nrcc_133_171.txt
