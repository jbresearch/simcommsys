# $Id$
# Library makefile

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
LIBRARY := $(notdir $(CURDIR))
LIBRARY := $(LIBRARY:Lib%=$(BUILDDIR)/lib%.a)

# Master targets

default:
	@echo No default target.

build:	$(LIBRARY)

install:	build

clean:
	@echo "Cleaning [$(TAG): $(RELEASE)]"
	@$(RM) $(BUILDDIR)

.PHONY:	default build install clean

# Manual targets

$(LIBRARY):	$(OBJECTS)
	@echo "Building library $* [$(TAG): $(RELEASE)]"
	@$(LIB) $(LIBflags) $(LIBRARY) $(OBJECTS)
	@$(RAN) $(LIBRARY)

# Pattern-matched targets

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
