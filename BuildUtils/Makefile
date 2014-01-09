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
# Utils binary makefile

## Note: this code should be independent, therefore:
# - Flags are reset to their defaults
# - Linking is not needed (single-source)
# - Dependencies are not needed
# - Libraries are not used

# Get list of source files
SOURCES := $(shell find . -name '*.cpp' -printf '%P\n')
CUDASRC := $(shell find . -name '*.cu' -printf '%P\n')
# Determine which of these we need to compile
ifeq ($(USE_CUDA),0)
TARGETS := $(SOURCES:%.cpp=bin/%)
else
TARGETS := $(SOURCES:%.cpp=bin/%) $(CUDASRC:%.cu=bin/%)
endif

## Reset flags

CCflags :=
NVCCflags :=
LDflags :=

## Commands

export MKDIR := mkdir -p
export NVCC := nvcc
export CC := gcc


# Master targets

default: build

build:	$(TARGETS)

clean:
	@echo "Cleaning"
	@$(RM) bin/*

.PHONY:	default build clean

# Pattern-matched targets

bin/%:	%.cu
	@$(MKDIR) $(dir $@)
	@echo "Compiling $<"
	@$(NVCC) $(NVCCflags) $< -o $@

bin/%:	%.cpp
	@$(MKDIR) $(dir $@)
	@echo "Compiling $<"
	@$(CC) $(CCflags) $< -o $@
