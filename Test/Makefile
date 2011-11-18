# Copyright (c) 2010 Johann A. Briffa
#
# This file is part of SimCommSys.
#
# SimCommSys is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SimCommSys is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
#
# $Id$
#
# Simulator test files makefile

SYSTEMS := $(wildcard Systems/*.txt)
SIMULATORS := $(SYSTEMS:Systems/%.txt=Simulators/%.txt)
TIMERS := $(SYSTEMS:Systems/%.txt=Timers/%.txt)

.force:

# main targets

all: $(SIMULATORS) $(TIMERS)

clean:
	@rm -f Simulators/*.txt Timers/*.txt

# GAWK programs

define GAWK_SIMULATOR
/^commsys/ {
	x=$$0
	switch(x) {
		# stream-oriented systems
		case /commsys_stream<.*/:
			sub(/commsys[^<]*/,"commsys_stream_simulator",x)
			break
		# all other matched objects
		default:
			sub(/commsys[^<]*/,"commsys_simulator",x)
			break
		}
	print x;
	}
{ print; }
endef
export GAWK_SIMULATOR

# pattern-matched targets

Simulators/%.txt: Systems/%.txt
	@echo Making $@
	@gawk "$$GAWK_SIMULATOR" < $< > $@

Timers/%.txt: Systems/%.txt
	@echo Making $@
	@gawk '/^commsys/ { x=$$0; sub(/commsys[^<]*/,"commsys_timer",x); print x; } { print; }' < $< > $@
