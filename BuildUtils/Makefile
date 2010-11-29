# $Id$
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
	@echo "Cleaning [$(TAG): $(RELEASE)]"
	@$(RM) $(BUILDDIR)

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
