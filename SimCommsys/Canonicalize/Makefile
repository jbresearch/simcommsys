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
# Target binary makefile

# All compiling, linking, and library flags are imported

# Get list of source files
SOURCES := $(shell find . -name '*.cpp' -printf '%P\n')
CUDASRC := $(shell find . -name '*.cu' -printf '%P\n')
# Determine which of these we need to compile
ifeq ($(USE_CUDA),0)
OBJECTS := $(SOURCES:%.cpp=$(BUILDDIR)/%.o)
else
OBJECTS := $(SOURCES:%.cpp=$(BUILDDIR)/%.o) $(CUDASRC:%.cu=$(BUILDDIR)/%.o)
endif
# Determine list of dependencies to create
DEPEND := $(OBJECTS:%.o=%.d)
# Automatically determine the final target name
TARGET := $(SOURCES:%.cpp=$(BUILDDIR)/%)
FINAL := $(SOURCES:%.cpp=$(BINDIR)/%.$(TAG).$(RELEASE))

# Master targets

default:
	@echo No default target.

build:	$(TARGET)

install:	$(FINAL)

clean:
	@echo "Cleaning [$(TAG): $(RELEASE)]"
	@$(RM) $(BUILDDIR)

.PHONY:	all install clean

# Manual targets

$(TARGET):	$(OBJECTS) $(LIBRARIES)
	@$(MKDIR) $(dir $@)
	@echo "Linking $(notdir $@) [$(TAG): $(RELEASE)]"
	@$(LD) -o $@ $(OBJECTS) $(LDflags)

# Pattern-matched targets

$(BINDIR)/%.$(TAG).$(RELEASE):	$(BUILDDIR)/%
	@$(MKDIR) $(dir $@)
	@echo "Installing $* [$(TAG): $(RELEASE)]"
	@$(CP) $< $@

$(BUILDDIR)/%.o:	%.cu
	@$(MKDIR) $(dir $@)
	@echo "Compiling $< [$(TAG): $(RELEASE)]"
	@$(NVCC) $(NVCCflags) -c $< -o $@

$(BUILDDIR)/%.o:	%.cpp
	@$(MKDIR) $(dir $@)
	@echo "Compiling $< [$(TAG): $(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	%.cu
	@$(MKDIR) $(dir $@)
	@echo "Making dependancy list for $*.o [$(TAG): $(RELEASE)]"
	@$(NVCC) $(NVCCflags) -M -odir $(dir $@) -o $@ $< 
	@sed -e 's,//,/,g' -e '\,/ , d' -e 's,\($*\)\.o[ :]*,\1.o $@ : ,g' -i $@

$(BUILDDIR)/%.d:	%.cpp
	@$(MKDIR) $(dir $@)
	@echo "Making dependancy list for $*.o [$(TAG): $(RELEASE)]"
	@$(CC) $(CCflags) -MM -MP -MT$(BUILDDIR)/$*.o -MF$@ $< 
	@sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' -i $@

# Dependency information

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPEND)
endif
