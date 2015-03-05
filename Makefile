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
# OpenMP Library (0 if absent)
ifndef USE_OMP
   export USE_OMP := 1
endif
# MPI Library (0 if absent)
ifndef USE_MPI
   export USE_MPI := $(shell mpic++ -showme 2>/dev/null |wc -l)
endif
# GMP Library (0 if absent)
ifndef USE_GMP
   TEST_GMP:=$(shell echo '\#include <gmp.h>' | g++ -H -E -o /dev/null - 2>&1 | grep error )
   export USE_GMP := $(if $(TEST_GMP),0,1)
endif
# CUDA compiler (0 if absent, architecture if present)
ifndef USE_CUDA
   export USE_CUDA := $(shell nvcc -V 2>/dev/null |wc -l)
   # Check for min supported architecture
   ifneq ($(USE_CUDA),0)
      # Compile tools if necessary
      ifeq (,$(wildcard BuildUtils/bin/getdevicearch))
         USE_CUDA := $(shell $(MAKE) -C "BuildUtils/" build)
      endif
      # Get the highest capability of installed cards
      USE_CUDA := $(shell BuildUtils/bin/getdevicearch 2>/dev/null)
      # If nothing was found
      ifeq (,$(USE_CUDA))
         USE_CUDA := 0
      endif
      # Ignore any pre-Fermi cards
      ifneq (,$(filter 10 11 12 13,$(USE_CUDA)))
         USE_CUDA := 0
      endif
   endif
endif
# Set default release to build
ifndef RELEASE
   export RELEASE := release
endif
# Validate release
ifneq ($(RELEASE),$(filter $(RELEASE),release debug profile))
   $(error Invalid release '$(RELEASE)')
endif


## Build version from git

export SIMCOMMSYS_VERSION := $(shell git describe --always --dirty)

## Build and installations details

# String to identify build
export BUILDID := $(notdir $(shell git rev-parse --abbrev-ref HEAD))
ifneq ($(USE_OMP),0)
   BUILDID := $(BUILDID)-omp
endif
ifneq ($(USE_MPI),0)
   BUILDID := $(BUILDID)-mpi
endif
ifneq ($(USE_GMP),0)
   BUILDID := $(BUILDID)-gmp
endif
ifneq ($(USE_CUDA),0)
   BUILDID := $(BUILDID)-cuda$(USE_CUDA)
endif

## Folders

# Root folder for package
export ROOTDIR := $(CURDIR)
# Folder for the build object files and binaries
export BUILDDIR = $(RELEASE)/$(OSARCH)/$(BUILDID)
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
   ifeq ($(MAKECMDGOALS),)
      ifneq ($(USE_OMP),0)
         $(info Using OMP: yes)
      endif
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
      $(info Build tag: $(BUILDID))
   endif
endif


## List of users libraries (in compilation order)

LIBNAMES := base image comm


## Commands

ifeq (,$(findstring no-print-directory,$(MAKEFLAGS)))
   export MAKE := $(MAKE) --no-print-directory
endif
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
LDopts := $(LDopts) -Wl,--start-group $(LIBNAMES:%=-l%) -Wl,--end-group
LDopts := $(LDopts) -lboost_program_options
# OMP options
ifneq ($(USE_OMP),0)
   LDopts := $(LDopts) -fopenmp
endif
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
# Standard libraries
LDopts := $(LDopts) -lm -lrt -lstdc++
# Debugging options
#LDopts := $(LDopts) -Wl,-v # show full ld command issued
#LDopts := $(LDopts) -Wl,-t # show full paths of linked objects
# release-dependent linking options
export LDflag_debug   := $(LDopts)
export LDflag_release := $(LDopts)
export LDflag_profile := -pg $(LDflag_release)
# Select the linking options to use
export LDflags = $(LDflag_$(RELEASE))


## Compiler settings

# Common options
CCopts := $(LIBNAMES:%=-I$(ROOTDIR)/Libraries/Lib%)
CCopts := $(CCopts) $(LIBNAMES:%=-I$(ROOTDIR)/Libraries/Lib%/$(BUILDDIR))
CCopts := $(CCopts) -Wall -Werror
#CCopts := $(CCopts) -std=c++0x
# note: below disabled to avoid problems with parallel builds
# note: below should be replaced with the following when we move to gcc > 4.4
#CCopts := $(CCopts) -save-temps
#CCopts := $(CCopts) -save-temps=obj
# OMP options
ifneq ($(USE_OMP),0)
   CCopts := $(CCopts) -DUSE_OMP -fopenmp
endif
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
      ifeq ($(OSARCH),ppc64)
         #CCopts := $(CCopts) -maltivec -m64
         CCopts := $(CCopts)
      else
         $(error Unknown architecture: $(OSARCH))
      endif
   endif
endif
# release-dependent compiler settings
export CCflag_debug := -g -DDEBUG $(CCopts)
export CCflag_release := -O3 -DNDEBUG $(CCopts)
export CCflag_profile := -pg $(CCflag_release)
# Select the compiler options to use
export CCflags = $(CCflag_$(RELEASE))


## CUDA Compiler settings

# Common options
NVCCopts := $(LIBNAMES:%=-I$(ROOTDIR)/Libraries/Lib%)
#NVCCopts := $(NVCCopts) -Xcompiler "-Wall,-Werror"
#NVCCopts := $(NVCCopts) -Xopencc "-woffall"
#NVCCopts := $(NVCCopts) -Xptxas "-v"
NVCCopts := $(NVCCopts) -w
NVCCopts := $(NVCCopts) -DUSE_CUDA
NVCCopts := $(NVCCopts) -arch=sm_$(USE_CUDA)
ifeq ($(OSARCH),i686)
   NVCCopts := $(NVCCopts) -m32
else
   ifeq ($(OSARCH),x86_64)
      NVCCopts := $(NVCCopts) -m64
   else
      ifeq ($(OSARCH),ppc64)
         NVCCopts := $(NVCCopts)
      else
         $(error Unknown architecture: $(OSARCH))
      endif
   endif
endif
# release-dependent compiler settings
NVCCflag_debug := -O0 -g -G -DDEBUG $(NVCCopts)
NVCCflag_release := -O3 -DNDEBUG $(NVCCopts)
NVCCflag_profile := -pg -DPROFILE $(NVCCflag_release)
# Select the compiler options to use
export NVCCflags := $(NVCCflag_$(RELEASE))


## Library builder settings

export LIBflags := ru


## User library list

export LIBRARIES = $(foreach name,$(LIBNAMES),$(ROOTDIR)/Libraries/Lib$(name)/$(BUILDDIR)/lib$(name).a)


### Local variables:

### Build Targets (libs is selective to avoid Libwin)

TARGETS_MAIN = $(wildcard SimCommsys/*)
TARGETS_TEST = $(wildcard Test/*)
TARGETS_LIBS = $(foreach name,$(LIBNAMES),/Libraries/Lib$(name))

## Master targets

default:
	@echo "No default target. General targets:"
	@echo "   <plain>-<cmd>-<set>-<release>"
	@echo "Where:"
	@echo "   <plain> = plain : disable optional libraries [optional]"
	@echo "   <cmd> = build|install|clean : build-only, install, or remove"
	@echo "   <set> = main|test|libs : what to build [default:main+test]"
	@echo "   <release> = debug|release|profile : [default:debug+release]"
	@echo "Master targets:"
	@echo "   all : equivalent to install and plain-install"
	@echo "   doc : compile code documentation"
	@echo "   clean-all : removes all binaries"
	@echo "   clean-dep : removes all dependency files"
	@echo "   showsettings : outputs compiler settings used"
	@echo "   buildid : outputs the build identifier string for the given settings"
	@echo "   version : outputs the version string"

all:
	@$(MAKE) install plain-install

doc:	FORCE
	@$(DOXYGEN)
#	@echo "*** To compile latex documentation: $(MAKE) -C doc/latex/"

clean-all:
	@echo "----> Cleaning all binaries."
	@find . -depth \( -name doc -or -name bin -or -iname debug -or -iname release -or -iname profile -or -name '*.s' -or -name '*.ii' -or -name '*.suo' -or -name '*.ncb' -or -name '*cache.dat' \) -print0 | xargs -0 rm -rf

clean-dep:
	@echo "----> Cleaning all dependency files."
	@find . -depth \( -name '*.d' \) -print0 | xargs -0 rm -rf

showsettings:
	$(CC) $(CCflag_release) -Q --help=target --help=optimizers --help=warnings

buildid:
	@echo $(BUILDID)

version:
	@echo $(SIMCOMMSYS_VERSION)

## Matched targets

plain-%:
	@$(MAKE) USE_OMP=0 USE_MPI=0 USE_GMP=0 USE_CUDA=0 $*

build:	build-main build-test
build-main:	build-main-debug build-main-release
build-test:	build-test-debug build-test-release
build-libs:	build-libs-debug build-libs-release

# libs build target is explicit here to avoid duplicate making
build-main-%:	build-libs-%
	@$(MAKE) RELEASE=$* DOTARGET=build $(TARGETS_MAIN)
build-test-%:	build-libs-%
	@$(MAKE) RELEASE=$* DOTARGET=build $(TARGETS_TEST)
build-libs-%:
	@$(MAKE) RELEASE=$* DOTARGET=build $(TARGETS_LIBS)

install:	install-main install-test
install-main:	install-main-debug install-main-release
install-test:	install-test-debug install-test-release
install-libs:	install-libs-debug install-libs-release

# libs install target is explicit here to avoid duplicate making
install-main-%:	install-libs-%
	@$(MAKE) RELEASE=$* DOTARGET=install $(TARGETS_MAIN)
install-test-%:	install-libs-%
	@$(MAKE) RELEASE=$* DOTARGET=install $(TARGETS_TEST)
install-libs-%:
	@$(MAKE) RELEASE=$* DOTARGET=install $(TARGETS_LIBS)

clean:	clean-main clean-test clean-libs
clean-main:	clean-main-release clean-main-debug
clean-test:	clean-test-release clean-test-debug
clean-libs:	clean-libs-release clean-libs-debug

clean-main-%:
	@$(MAKE) RELEASE=$* DOTARGET=clean $(TARGETS_MAIN)
clean-test-%:
	@$(MAKE) RELEASE=$* DOTARGET=clean $(TARGETS_TEST)
clean-libs-%:
	@$(MAKE) RELEASE=$* DOTARGET=clean $(TARGETS_LIBS)

## Setting targets

FORCE:

.PHONY:	all build install clean showsettings tag

.SUFFIXES: # Delete the default suffixes

.DELETE_ON_ERROR:

## Manual targets

$(TARGETS_MAIN) $(TARGETS_TEST):	$(TARGETS_LIBS) FORCE
	@echo "----> Making target \"$(notdir $@)\" [$(BUILDID): $(RELEASE)]."
	@$(MAKE) -C "$(ROOTDIR)/$@" $(DOTARGET)

$(TARGETS_LIBS):	FORCE
	@echo "----> Making library \"$(notdir $@)\" [$(BUILDID): $(RELEASE)]."
	@$(MAKE) -C "$(ROOTDIR)/$@" $(DOTARGET)
