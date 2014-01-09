#!/bin/bash

# first make sure we have all systems set up
make all

echo "Quick simulation with 'uncoded' codec..."
quicksimulation.master.release -t 10 -r 6.8 -i Simulators/errors_hamming-random-awgn-bpsk-uncoded.txt >Results/sim.errors_hamming-random-awgn-bpsk-uncoded.txt 2>/dev/null

echo "Quick simulation with concatenated RS-convolutional codecs..."
quicksimulation.master.release -t 10 -r 2.2 -i Simulators/errors_hamming-random-awgn-bpsk-concantenated-reedsolomon_255_223_gf256-map_interleaved-nrcc_133_171.txt >Results/sim.errors_hamming-random-awgn-bpsk-concantenated-reedsolomon_255_223_gf256-map_interleaved-nrcc_133_171.txt 2>/dev/null
