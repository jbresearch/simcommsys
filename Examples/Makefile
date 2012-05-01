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

# List of symbol types to work with
STREAM_LIMITS := 1 10 100 1000 10000

# List of systems to work with
SYSTEMS := $(wildcard Systems/*.txt)
SIMULATORS := $(SYSTEMS:Systems/%.txt=Simulators/%.txt) \
	$(foreach A,$(STREAM_LIMITS),$(SYSTEMS:Systems/stream-%.txt=Simulators/stream-$(A)-%.txt))
TIMERS := $(SYSTEMS:Systems/%.txt=Timers/%.txt)

.force:

# main targets

all: $(SIMULATORS) $(TIMERS)

clean:
	@rm -f Simulators/*.txt Timers/*.txt

# GAWK programs

define GAWK_SIMULATOR_CREATOR
define GAWK_SIMULATOR_$(1)
/^commsys_stream/ {
	x=$$$$0
	# stream-oriented systems
	sub(/commsys[^<]*/,"commsys_stream_simulator",x)
	# add levenshtein-distance error rate
	x = gensub(/<(.+)>/,"<\\1,levenshtein>","g",x)
	print x;
	# add version and stream limit
	print 1
	print $(1)
	}
/^commsys/ && !/^commsys_stream/ {
	x=$$$$0
	# all other matched objects
	sub(/commsys[^<]*/,"commsys_simulator",x)
	# add levenshtein-distance error rate
	x = gensub(/<(.+)>/,"<\\1,levenshtein>","g",x)
	print x;
	}
{ print; }
endef
export GAWK_SIMULATOR_$(1)
endef

$(eval $(call GAWK_SIMULATOR_CREATOR,0))

$(foreach A,$(STREAM_LIMITS),\
	$(eval $(call GAWK_SIMULATOR_CREATOR,$(A))) )

define GAWK_TIMER
/^commsys/ {
	x=$$0
	sub(/commsys[^<]*/,"commsys_timer",x)
	print x;
	}
{ print; }
endef
export GAWK_TIMER

# pattern-matched targets

define GenerateSimulators
Simulators/stream-$(1)-%.txt: Systems/stream-%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_$(1)" < $$< > $$@
endef

$(foreach A,$(STREAM_LIMITS),\
	$(eval $(call GenerateSimulators,$(A))) )

Simulators/%.txt: Systems/%.txt
	@echo Making $@
	@gawk "$$GAWK_SIMULATOR_0" < $< > $@

Timers/%.txt: Systems/%.txt
	@echo Making $@
	@gawk "$$GAWK_TIMER" < $< > $@
