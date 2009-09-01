# (c) 2009: Johann Briffa <j.briffa@ieee.org>
# $Id$

SYSTEMS := $(wildcard Systems/*.txt)
SIMULATORS := $(SYSTEMS:Systems/%.txt=Simulators/%.txt)

.force:

all: $(SIMULATORS)

Simulators/%.txt: Systems/%.txt
	@echo Making $@
	@gawk '/^commsys/ { x=$$0; sub(/commsys[^<]*/,"commsys_simulator",x); print x; } { print; }' < $< > $@
