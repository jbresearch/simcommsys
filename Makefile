# $Author$
# $Revision$
# $Date$

# Exported variables:

# Root folder for package
export ROOTDIR := $(PWD)

# Directory where the object files and binaries are placed
export OSNAME := $(shell uname -s)
export OSARCH := $(shell uname -m)
export BUILDDIR = Arch/$(OSNAME).$(OSARCH)/$(RELEASE)
export BINDIR = ~/bin.$(OSARCH)

# Linker settings
export LDlibusr := -lcomm -lbase
LDlibsys := -lm -lstdc++
#LDlibsys := -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic -lm
#LDlibmpi := `mpic++ -showme:link`
#LDlibmpi := -lpmpich++ -lmpich
LDlibmpi :=
export LDlibs := $(LDlibusr) $(LDlibsys) $(LDlibmpi)
# Define linking flags
export LDflagProfile := -pg
export LDflagRelease := 
export LDflagDebug   := 
LDflagsCommon :=
#LDflagsCommon := -static-libgcc
export LDflags = $(LDflagsCommon) $(LDflag$(RELEASE)) $(LDlibusr:-l%=-L$(ROOTDIR)/Libraries/Lib%/$(BUILDDIR))

# Compiler settings
CCprfopt := -pg -O3 -DNDEBUG
CCrelopt := -O3 -DNDEBUG
CCdbgopt := -g -DDEBUG
CClibs := $(LDlibusr:-l%=-I$(ROOTDIR)/Libraries/Lib%/Source)
CClang := -Wall -Werror -Wno-non-template-friend -Woverloaded-virtual
#CCmpi := -DUSEMPI `mpic++ -showme:compile`
#CCmpi := -DUSEMPI -DUSE_STDARG -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STDARG_H=1 -DUSE_STDARG=1 -DMALLOC_RET_VOID=1
CCmpi :=
# Define compiling flags
export CCflagProfile := $(CCprfopt) $(CClibs) $(CClang) $(CCmpi)
export CCflagRelease := $(CCrelopt) $(CClibs) $(CClang) $(CCmpi)
export CCflagDebug   := $(CCdbgopt) $(CClibs) $(CClang) $(CCmpi)
export CCflags = $(CCflag$(RELEASE))
export CCdepend := -MM

# User library list
export LIBS = $(foreach name,$(LDlibusr:-l%=%),$(ROOTDIR)/Libraries/Lib$(name)/$(BUILDDIR)/lib$(name).a)

# Library builder settings
export LIBflags := ru

# Define the names for commands
export MKDIR := mkdir -p
export RM := rm -rf
export CP := cp
export CC := gcc
export LD := gcc
export LIB := ar
export RAN := ranlib

# Still to define down here - BUILDDIR, CCflags ##


# Local variables:

TARGETS := Turbo\ Codes

# Master targets

all:     debug release profile

profile:
	@$(MAKE) RELEASE=Profile DOTARGET=all $(TARGETS)

release:
	@$(MAKE) RELEASE=Release DOTARGET=all $(TARGETS)

debug:
	@$(MAKE) RELEASE=Debug DOTARGET=all $(TARGETS)

install:
	@$(MAKE) RELEASE=Release DOTARGET=install $(TARGETS)

clean:

FORCE:

.PHONY:	all debug release profile install clean

# Manual targets

$(TARGETS):	$(LIBS) FORCE
	@echo "----> Making target \"$(notdir $@)\" [$(RELEASE)]."
	@$(MAKE) -C "$(ROOTDIR)/$@" $(DOTARGET)

# Pattern-matched targets

%.a:	FORCE
	@echo "----> Making library \"$(notdir $@)\" [$(RELEASE)]."
	@$(MAKE) -C $(ROOTDIR)/Libraries/$(patsubst lib%.a,Lib%,$(notdir $@))
