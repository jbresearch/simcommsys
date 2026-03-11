# Introductory Paper

This directory contains the system file examples illustrated in the [original paper on Simcommsys](https://jabriffa.wordpress.com/2014/06/30/journal-iet-journal-of-engineering-2/).

## Building simulator and timer files

The first step in running any Simcommsys simulation is to build simulator/timer files from the base "system files" which describe a communication system. The system files are kept in `Systems/`, while the simulator/timer files are kept under `Simulators/`, `Timers/` respectively.

In order to build these files, run:

```
make TAG=<build-tag>
```

where tag is the binary tag for your Simcommsys build. If you are building Simcommsys from source you can find this by running:

```
make showversion
make showbuildid
```

from the top-level directory of the Simcommsys repo.

## Running simulations

In order to run simulations locally, you can use:

```
make run-jobfile-local
```
