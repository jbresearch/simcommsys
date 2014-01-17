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
# Simulator files makefile

# List of stream lengths to work with for reset and termination respectively
STREAM_RESET_N := 1
STREAM_TERM_N := 1
# List of results collectors to work with
COLLECTOR_TYPES := errors_hamming
# List of input stream modes
INPUT_TYPES := random

# List of systems to work with
SYSTEMS := $(wildcard Systems/*.txt)
SIMULATORS := $(foreach C,$(INPUT_TYPES), \
	$(foreach B,$(COLLECTOR_TYPES), \
	$(SYSTEMS:Systems/%.txt=Simulators/$(B)-$(C)-%.txt) \
	$(foreach A,$(STREAM_RESET_N),$(SYSTEMS:Systems/stream-%.txt=Simulators/$(B)-$(C)-stream-reset-$(A)-%.txt)) \
	$(foreach A,$(STREAM_TERM_N),$(SYSTEMS:Systems/stream-%.txt=Simulators/$(B)-$(C)-stream-term-$(A)-%.txt)) \
	))
TIMERS := $(SYSTEMS:Systems/%.txt=Timers/%.txt)

# main targets

default:
	@echo "No default target. Choose from:"
	@echo "   all : equivalent to simulators and timers"
	@echo "   simulators : create a simulator for each system"
	@echo "   timers : create a timer sim for each system"
	@echo "   clean : removes all generated files"

all: simulators timers
simulators: $(SIMULATORS)
timers: $(TIMERS)

clean:
	@rm -f Simulators/*.txt Timers/*.txt

## Setting targets

FORCE:

.PHONY:	all simulators timers clean

.SUFFIXES: # Delete the default suffixes

.DELETE_ON_ERROR:

## GAWK programs

define GAWK_SETUP
	# setup
	mode["zero"] = 0
	mode["random"] = 1
	mode["user"] = 2
endef

define GAWK_USERSEQ
	# output user sequence
	print 1
endef

# Simulators with parameters: streams with reset

define GAWK_SIMULATOR_STREAM_RESET_CREATOR
define GAWK_SIMULATOR_STREAM_RESET_$(1)_$(2)_$(3)
# stream-oriented systems
/^commsys_stream/ {
	$(GAWK_SETUP)
	x=$$$$0
	# use stream simulator
	sub(/commsys[^<]*/,"commsys_stream_simulator",x)
	# remove container type and add results collector
	x = gensub(/<(.+),vector>/,"<\\1,$(2)>","g",x)
	print x;
	# add stream-simulator version, streaming mode, and corresponding length
	print "# Version"
	print 2
	print "# Streaming mode (0=open, 1=reset, 2=terminated)"
	print 1
	print "# Number of frames to reset"
	print $(1)
	# add simulator version, and input mode
	print "# Version"
	print 2
	print "# Input mode (0=zero, 1=random, 2=user[seq])"
	print mode["$(3)"]
	if ("$(3)" == "user")
		{
		print "#: input symbols - count"
		print 1
		print "#: input symbols - values"
		$(GAWK_USERSEQ)
		}
	print "# Communication system"
	}
{ print; }
endef
export GAWK_SIMULATOR_STREAM_RESET_$(1)_$(2)_$(3)
endef

$(foreach C,$(INPUT_TYPES),\
	$(foreach B,$(COLLECTOR_TYPES),\
		$(foreach A,$(STREAM_RESET_N),\
			$(eval $(call GAWK_SIMULATOR_STREAM_RESET_CREATOR,$(A),$(B),$(C))) )))

# Simulators with parameters: streams with termination

define GAWK_SIMULATOR_STREAM_TERM_CREATOR
define GAWK_SIMULATOR_STREAM_TERM_$(1)_$(2)_$(3)
# stream-oriented systems
/^commsys_stream/ {
	$(GAWK_SETUP)
	x=$$$$0
	# use stream simulator
	sub(/commsys[^<]*/,"commsys_stream_simulator",x)
	# remove container type and add results collector
	x = gensub(/<(.+),vector>/,"<\\1,$(2)>","g",x)
	print x;
	# add stream-simulator version, streaming mode, and corresponding length
	print "# Version"
	print 2
	print "# Streaming mode (0=open, 1=reset, 2=terminated)"
	print 2
	print "# Length of stream in frames"
	print $(1)
	# add simulator version, and input mode
	print "# Version"
	print 2
	print "# Input mode (0=zero, 1=random, 2=user[seq])"
	print mode["$(3)"]
	if ("$(3)" == "user")
		{
		print "#: input symbols - count"
		print 1
		print "#: input symbols - values"
		$(GAWK_USERSEQ)
		}
	print "# Communication system"
	}
{ print; }
endef
export GAWK_SIMULATOR_STREAM_TERM_$(1)_$(2)_$(3)
endef

$(foreach C,$(INPUT_TYPES),\
	$(foreach B,$(COLLECTOR_TYPES),\
		$(foreach A,$(STREAM_TERM_N),\
			$(eval $(call GAWK_SIMULATOR_STREAM_TERM_CREATOR,$(A),$(B),$(C))) )))

# Simulators without parameters: standard commsys and open-ended streams

define GAWK_SIMULATOR_CREATOR
define GAWK_SIMULATOR_$(1)_$(2)
# stream-oriented systems
/^commsys_stream/ {
	$(GAWK_SETUP)
	x=$$$$0
	# use stream simulator
	sub(/commsys[^<]*/,"commsys_stream_simulator",x)
	# remove container type and add results collector
	x = gensub(/<(.+),vector>/,"<\\1,$(1)>","g",x)
	print x;
	# add stream-simulator version and open-ended streaming mode
	print "# Version"
	print 2
	print "# Streaming mode (0=open, 1=reset, 2=terminated)"
	print 0
	# add simulator version, and input mode
	print "# Version"
	print 2
	print "# Input mode (0=zero, 1=random, 2=user[seq])"
	print mode["$(2)"]
	if ("$(2)" == "user")
		{
		print "#: input symbols - count"
		print 1
		print "#: input symbols - values"
		$(GAWK_USERSEQ)
		}
	print "# Communication system"
	}
# all other matched systems
/^commsys/ && !/^commsys_stream/ {
	$(GAWK_SETUP)
	x=$$$$0
	# use standard simulator
	sub(/commsys[^<]*/,"commsys_simulator",x)
	# remove container type and add results collector
	x = gensub(/<(.+),vector>/,"<\\1,$(1)>","g",x)
	print x;
	# add simulator version, and input mode
	print "# Version"
	print 2
	print "# Input mode (0=zero, 1=random, 2=user[seq])"
	print mode["$(2)"]
	if ("$(2)" == "user")
		{
		print "#: input symbols - count"
		print 1
		print "#: input symbols - values"
		$(GAWK_USERSEQ)
		}
	print "# Communication system"
	}
{ print; }
endef
export GAWK_SIMULATOR_$(1)_$(2)
endef

$(foreach B,$(INPUT_TYPES),\
	$(foreach A,$(COLLECTOR_TYPES),\
		$(eval $(call GAWK_SIMULATOR_CREATOR,$(A),$(B))) ))

# Timer simulations

define GAWK_TIMER
# all matched systems
/^commsys/ {
	x=$$0
	# use timer simulator
	sub(/commsys[^<]*/,"commsys_timer",x)
	# remove container type
	x = gensub(/<(.+),vector>/,"<\\1>","g",x)
	print x;
	# add simulator version, and input mode
	print "# Version"
	print 2
	print "# Input mode (0=zero, 1=random, 2=user[seq])"
	print 1
	print "# Communication system"
	}
{ print; }
endef
export GAWK_TIMER

## pattern-matched targets

# standard commsys and open-ended streams
define GenerateSimulators
Simulators/$(1)-$(2)-%.txt: Systems/%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_$(1)_$(2)" < $$< > $$@
endef

$(foreach B,$(INPUT_TYPES),\
	$(foreach A,$(COLLECTOR_TYPES),\
		$(eval $(call GenerateSimulators,$(A),$(B))) ))

# streams with reset
define GenerateSimulatorsStreamReset
Simulators/$(2)-$(3)-stream-reset-$(1)-%.txt: Systems/stream-%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_STREAM_RESET_$(1)_$(2)_$(3)" < $$< > $$@
endef

$(foreach C,$(INPUT_TYPES),\
	$(foreach B,$(COLLECTOR_TYPES),\
		$(foreach A,$(STREAM_RESET_N),\
			$(eval $(call GenerateSimulatorsStreamReset,$(A),$(B),$(C))) )))

# streams with termination
define GenerateSimulatorsStreamTerm
Simulators/$(2)-$(3)-stream-term-$(1)-%.txt: Systems/stream-%.txt
	@echo Making $$@
	@gawk "$$$$GAWK_SIMULATOR_STREAM_TERM_$(1)_$(2)_$(3)" < $$< > $$@
endef

$(foreach C,$(INPUT_TYPES),\
	$(foreach B,$(COLLECTOR_TYPES),\
		$(foreach A,$(STREAM_TERM_N),\
			$(eval $(call GenerateSimulatorsStreamTerm,$(A),$(B),$(C))) )))

Timers/%.txt: Systems/%.txt
	@echo Making $@
	@gawk "$$GAWK_TIMER" < $< > $@
