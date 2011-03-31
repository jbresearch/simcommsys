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
# Master makefile

### Exported variables:


## Control variables

# Build Architecture (ie i686 x86_64).
ifndef OSARCH
export OSARCH := $(shell uname -m)
endif
# Kernel version
ifndef KERNEL
export KERNEL := $(shell uname -r)
endif
# Number of CPUs
ifndef CPUS
export CPUS := $(shell grep processor /proc/cpuinfo |wc -l)
endif
# MPI Library (0 if absent)
ifndef USE_MPI
export USE_MPI := $(shell mpic++ -showme 2>/dev/null |wc -l)
endif
# GMP Library (0 if absent)
ifndef USE_GMP
export USE_GMP := $(if $(wildcard /usr/include/gmp.h),1,0)
endif
# CUDA compiler (0 if absent, architecture if present)
ifndef USE_CUDA
export USE_CUDA := $(shell nvcc -V 2>/dev/null |wc -l)
endif
ifneq ($(USE_CUDA),0)
ifeq (,$(wildcard BuildUtils/bin/getdevicearch))
USE_CUDA := $(shell $(MAKE) -C "BuildUtils/" build)
endif
USE_CUDA := $(shell BuildUtils/bin/getdevicearch 2>/dev/null)
ifeq (,$(USE_CUDA))
USE_CUDA := 10
endif
endif
# Set default release to build
ifndef RELEASE
export RELEASE := Release
endif


## Build and installations details

# Tag to identify build
export TAG := $(notdir $(CURDIR))
ifneq ($(USE_MPI),0)
TAG := $(TAG)-mpi
endif
ifneq ($(USE_GMP),0)
TAG := $(TAG)-gmp
endif
ifneq ($(USE_CUDA),0)
TAG := $(TAG)-cuda$(USE_CUDA)
endif

## Folders

# Root folder for package
export ROOTDIR := $(CURDIR)
# Folder for the build object files and binaries
export BUILDDIR = $(RELEASE)/$(OSARCH)/$(TAG)
# Folder for installed binaries
ifndef BINDIR
ifeq ($(shell [ -d ~/bin.$(KERNEL) ] && echo 1),1)
export BINDIR = ~/bin.$(KERNEL)
else
export BINDIR = ~/bin.$(OSARCH)
endif
else
export BINDIR
endif

## User pacifier
ifeq ($(MAKELEVEL),0)
ifneq ($(USE_MPI),0)
$(info Using MPI: yes)
endif
ifneq ($(USE_GMP),0)
$(info Using GMP: yes)
endif
ifneq ($(USE_CUDA),0)
$(info Using CUDA: yes, compute model $(USE_CUDA))
endif
$(info Install folder: $(BINDIR))
$(info Build tag: $(TAG))
endif

## Version control information

WCURL := $(shell svn info |gawk '/^URL/ { print $$2 }')
WCVER := $(shell svnversion)


## List of users libraries (in linking order)

LIBNAMES := comm image base


## Commands

export MAKE := $(MAKE) --no-print-directory
export MKDIR := mkdir -p
export RM := rm -rf
export CP := cp
export NVCC := nvcc
export CC := gcc
export LD := gcc
export LIB := ar
export RAN := ranlib
export DOXYGEN := doxygen


## Linker settings

# Common options
LDopts := $(LIBNAMES:%=-L$(ROOTDIR)/Libraries/Lib%/$(BUILDDIR))
LDopts := $(LDopts) $(LIBNAMES:%=-l%)
LDopts := $(LDopts) -lm -lstdc++ -lboost_program_options
# MPI options
ifneq ($(USE_MPI),0)
LDopts := $(LDopts) $(shell mpic++ -showme:link)
endif
# GMP options
ifneq ($(USE_GMP),0)
LDopts := $(LDopts) -lgmpxx -lgmp
endif
# CUDA options
ifneq ($(USE_CUDA),0)
ifeq ($(OSARCH),x86_64)
LDopts := $(LDopts) -L/usr/local/cuda/lib64 -lcudart
else
LDopts := $(LDopts) -L/usr/local/cuda/lib -lcudart
endif
endif
# Release-dependent linking options
export LDflagDebug   := $(LDopts)
export LDflagRelease := $(LDopts)
export LDflagProfile := -pg $(LDflagRelease)
# Select the linking options to use
export LDflags = $(LDflag$(RELEASE))


## Compiler settings

# Common options
CCopts := $(LIBNAMES:%=-I$(ROOTDIR)/Libraries/Lib%)
CCopts := $(CCopts) -Wall -Werror
CCopts := $(CCopts) -D__WCVER__=\"$(WCVER)\" -D__WCURL__=\"$(WCURL)\"
# MPI options
ifneq ($(USE_MPI),0)
CCopts := $(CCopts) -DUSE_MPI $(shell mpic++ -showme:compile)
endif
# GMP options
ifneq ($(USE_GMP),0)
CCopts := $(CCopts) -DUSE_GMP
endif
# CUDA options
ifneq ($(USE_CUDA),0)
CCopts := $(CCopts) -DUSE_CUDA
endif
# Architecture-specific options
ifeq ($(OSARCH),i686)
CCopts := $(CCopts) -msse2
else
ifeq ($(OSARCH),x86_64)
CCopts := $(CCopts) -msse2 -m64
else
$(error Unknown architecture: $(OSARCH))
endif
endif
# Release-dependent compiler settings
export CCflagDebug := -g -DDEBUG $(CCopts)
export CCflagRelease := -O3 -DNDEBUG $(CCopts)
export CCflagProfile := -pg $(CCflagRelease)
# Select the compiler options to use
export CCflags = $(CCflag$(RELEASE))


## CUDA Compiler settings

# Common options
NVCCopts := $(LIBNAMES:%=-I$(ROOTDIR)/Libraries/Lib%)
#NVCCopts := $(NVCCopts) -Xcompiler "-Wall,-Werror"
NVCCopts := $(NVCCopts) -Xopencc "-woffall"
NVCCopts := $(NVCCopts) -D__WCVER__=\"$(WCVER)\" -D__WCURL__=\"$(WCURL)\"
NVCCopts := $(NVCCopts) -DUSE_CUDA
NVCCopts := $(NVCCopts) -arch=compute_$(USE_CUDA) -code=compute_$(USE_CUDA),sm_$(USE_CUDA)
ifeq ($(OSARCH),i686)
NVCCopts := $(NVCCopts) -m32
else
ifeq ($(OSARCH),x86_64)
NVCCopts := $(NVCCopts) -m64
else
$(error Unknown architecture: $(OSARCH))
endif
endif
# Release-dependent compiler settings
NVCCflagDebug := -O0 -g -G -DDEBUG $(NVCCopts)
NVCCflagRelease := -O3 -DNDEBUG $(NVCCopts)
NVCCflagProfile := -pg -DPROFILE $(NVCCflagRelease)
# Select the compiler options to use
export NVCCflags := $(NVCCflag$(RELEASE))


## Library builder settings

export LIBflags := ru


## User library list

export LIBRARIES = $(foreach name,$(LIBNAMES),$(ROOTDIR)/Libraries/Lib$(name)/$(BUILDDIR)/lib$(name).a)


### Local variables:

TARGETS = $(wildcard SimCommsys/*)


### Build Targets


## Master targets

default:
	@echo No default target.

all:
	@$(MAKE) -j$(CPUS) install
	@$(MAKE) -j$(CPUS) plain-install

plain-build:
	@$(MAKE) USE_CUDA=0 USE_MPI=0 USE_GMP=0 build

plain-install:
	@$(MAKE) USE_CUDA=0 USE_MPI=0 USE_GMP=0 install

build:     debug release

profile:
	@$(MAKE) RELEASE=Profile DOTARGET=build $(TARGETS)

release:
	@$(MAKE) RELEASE=Release DOTARGET=build $(TARGETS)

debug:
	@$(MAKE) RELEASE=Debug DOTARGET=build $(TARGETS)

install:	install-release install-debug

install-profile:
	@$(MAKE) RELEASE=Profile DOTARGET=install $(TARGETS)

install-release:
	@$(MAKE) RELEASE=Release DOTARGET=install $(TARGETS)

install-debug:
	@$(MAKE) RELEASE=Debug DOTARGET=install $(TARGETS)

doc:
	@$(DOXYGEN)

clean:
	@$(MAKE) RELEASE=Debug DOTARGET=clean $(TARGETS)
	@$(MAKE) RELEASE=Release DOTARGET=clean $(TARGETS)
	@$(MAKE) RELEASE=Profile DOTARGET=clean $(TARGETS)

clean-all:
	find . -depth \( -name doc -or -name Debug -or -name Release -or -name Profile -or -name '*.suo' -or -name '*.ncb' -or -name '*cache.dat' \) -exec rm -rf '{}' \;

showsettings:
	$(CC) $(CCflagRelease) -Q --help=target --help=optimizers --help=warnings

FORCE:

.PHONY:	all debug release profile install clean


## Manual targets

$(TARGETS):	$(LIBRARIES) FORCE
	@echo "----> Making target \"$(notdir $@)\" [$(TAG): $(RELEASE)]."
	@$(MAKE) -C "$(ROOTDIR)/$@" $(DOTARGET)

BuildUtils:	FORCE
	@echo "----> Making target \"$@\"."
	@$(MAKE) -C "$(ROOTDIR)/$@" build

## Pattern-matched targets

%.a:	FORCE
	@echo "----> Making library \"$(notdir $@)\" [$(TAG): $(RELEASE)]."
	@$(MAKE) -C $(ROOTDIR)/Libraries/$(patsubst lib%.a,Lib%,$(notdir $@)) $(DOTARGET)
