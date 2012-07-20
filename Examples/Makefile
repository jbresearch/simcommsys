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
#STREAM_LIMITS := 1 10 100 1000 10000
STREAM_LIMITS := 10
# List of results collectors to work with
COLLECTOR_TYPES := errors_levenshtein prof_pos

# Set default tag and release to test with
ifndef TAG
export TAG := $(notdir $(shell cd .. && pwd))
endif
ifndef RELEASE
export RELEASE := debug
endif

# List of systems to work with
SYSTEMS := $(wildcard Systems/*.txt)
SIMULATORS := $(foreach B,$(COLLECTOR_TYPES), $(SYSTEMS:Systems/%.txt=Simulators/$(B)-%.txt) \
	$(foreach A,$(STREAM_LIMITS),$(SYSTEMS:Systems/stream-%.txt=Simulators/$(B)-stream-$(A)-%.txt)))
TIMERS := $(SYSTEMS:Systems/%.txt=Timers/%.txt)

# main targets

default:
	@echo "No default target. Choose from:"
	@echo "   all : equivalent to simulators and timers"
	@echo "   simulators : create a simulator for each system"
	@echo "   timers : create a timer sim for each system"
	@echo "   test-all : equivalent to test-simulators and test-timers"
	@echo "   test-simulators : create a test log for each simulator"
	@echo "   test-timers : create a test log for each timer"
	@echo "   clean : removes all generated files"

all: simulators timers
simulators: $(SIMULATORS)
timers: $(TIMERS)

test-all: test-simulators test-timers
test-simulators: $(SIMULATORS:Simulators/%.txt=TestLogs/sim.%.txt)
test-timers: $(TIMERS:Timers/%.txt=TestLogs/tim.%.txt)

clean:
	@rm -f Simulators/*.txt Timers/*.txt TestLogs/*.txt

## Setting targets

FORCE:

.PHONY:	all simulators timers test-all test-simulators test-timers clean

.SUFFIXES: # Delete the default suffixes

.DELETE_ON_ERROR:

# GAWK programs

define GAWK_SIMULATOR_CREATOR
define GAWK_SIMULATOR_$(1)_$(2)
/^commsys_stream/ {
	x=$$$$0
	# stream-oriented systems
	sub(/commsys[^<]*/,"commsys_stream_simulator",x)
	# remove container type and
	# add levenshtein-distance error rate
	x = gensub(/<(.+),vector>/,"<\\1,$(2)>","g",x)
	print x;
	# add version and stream limit
	print 1
	print $(1)
	}
/^commsys/ && !/^commsys_stream/ {
	x=$$$$0
	# all other matched objects
	sub(/commsys[^<]*/,"commsys_simulator",x)
        # remove container type and
        # add levenshtein-distance error rate
	x = gensub(/<(.+),vector>/,"<\\1,$(2)>","g",x)
	print x;
	}
{ print; }
endef
export GAWK_SIMULATOR_$(1)_$(2)
endef

$(foreach B,0 $(COLLECTOR_TYPES),\
	$(foreach A,0 $(STREAM_LIMITS),\
		$(eval $(call GAWK_SIMULATOR_CREATOR,$(A),$(B))) ))

define GAWK_TIMER
/^commsys/ {
	x=$$0
	# all matched objects
	sub(/commsys[^<]*/,"commsys_timer",x)
	# remove container type
	x = gensub(/<(.+),vector>/,"<\\1>","g",x)
	print x;
	}
{ print; }
endef
export GAWK_TIMER

# pattern-matched targets

define GenerateSimulators
Simulators/$(2)-%.txt: Systems/%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_0_$(2)" < $$< > $$@

Simulators/$(2)-stream-$(1)-%.txt: Systems/stream-%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_$(1)_$(2)" < $$< > $$@
endef

$(foreach B,0 $(COLLECTOR_TYPES),\
	$(foreach A,0 $(STREAM_LIMITS),\
		$(eval $(call GenerateSimulators,$(A),$(B))) ))

Timers/%.txt: Systems/%.txt
	@echo Making $@
	@gawk "$$GAWK_TIMER" < $< > $@

TestLogs/sim.%.txt: Simulators/%.txt
	@echo Testing $<
	@SPECturbo.$(TAG).$(RELEASE) -t 1 -r 1e-2 -i $< > $@ 2>&1

TestLogs/tim.%.txt: Timers/%.txt
	@echo Testing $<
	@SPECturbo.$(TAG).$(RELEASE) -t 1 -r 1e-2 -i $< > $@ 2>&1
