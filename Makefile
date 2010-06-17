# $Author$
# $Revision$
# $Date$

# Exported variables:

# Root folder for package
export ROOTDIR := $(PWD)

# Directory where the object files and binaries are placed
# if OSARCH is available then use it - otherwise use the
# processor architecture (ie i686 x86_64)
ifndef OSARCH
export OSARCH := $(shell uname -m)
else
export OSARCH
endif
export BUILDDIR = $(RELEASE)/$(OSARCH)
export BINDIR = ~/bin.$(OSARCH)

# Version control information
WCURL := $(shell svn info |gawk '/^URL/ { print $$2 }')
WCVER := $(shell svnversion)
export WCTAG := $(notdir $(PWD))

# Linker settings
export LDlibusr := -limage -lcomm -lbase
LDlibsys := -lm -lstdc++ -lboost_program_options
#LDlibmpi := `mpic++ -showme:link`
#LDlibmpi := -lpmpich++ -lmpich
LDlibmpi :=
export LDlibs := $(LDlibusr) $(LDlibsys) $(LDlibmpi)
# Define linking flags
export LDflagProfile := -pg
export LDflagRelease := 
export LDflagDebug   := 
#LDflagsCommon := -static-libgcc
LDflagsCommon :=
export LDflags = $(LDflagsCommon) $(LDflag$(RELEASE)) $(LDlibusr:-l%=-L$(ROOTDIR)/Libraries/Lib%/$(BUILDDIR))

# Target-specific common options
CCcomopt :=
ifeq ($(OSARCH),i686)
CCcomopt := -msse2 $(CCcomopt)
else ifeq ($(OSARCH),x86_64)
CCcomopt := -msse2 -m64 $(CCcomopt)
endif
# Compiler settings
CCdbgopt := -g -DDEBUG $(CCcomopt)
CCrelopt := -O3 -DNDEBUG $(CCcomopt)
#CCprfopt := $(CCrelopt) -pg -fno-inline
CCprfopt := $(CCrelopt) -pg
CClibs := $(LDlibusr:-l%=-I$(ROOTDIR)/Libraries/Lib%)
#CClang := -Wall -Werror -Wno-non-template-friend -Woverloaded-virtual
CClang := -Wall -Werror
#CCmpi := -DUSEMPI `mpic++ -showme:compile`
#CCmpi := -DUSEMPI -DUSE_STDARG -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STDARG_H=1 -DUSE_STDARG=1 -DMALLOC_RET_VOID=1
CCmpi :=
CCsvn := -D__WCVER__=\"$(WCVER)\" -D__WCURL__=\"$(WCURL)\"
# Define compiling flags
export CCflagProfile := $(CCprfopt) $(CClibs) $(CClang) $(CCmpi) $(CCsvn)
export CCflagRelease := $(CCrelopt) $(CClibs) $(CClang) $(CCmpi) $(CCsvn)
export CCflagDebug   := $(CCdbgopt) $(CClibs) $(CClang) $(CCmpi) $(CCsvn)
export CCflags = $(CCflag$(RELEASE))
export CCdepend := -MM -MP

# User library list
export LIBS = $(foreach name,$(LDlibusr:-l%=%),$(ROOTDIR)/Libraries/Lib$(name)/$(BUILDDIR)/lib$(name).a)

# Library builder settings
export LIBflags := ru

# Define the names for commands
export MAKE := $(MAKE) --no-print-directory
export MKDIR := mkdir -p
export RM := rm -rf
export CP := cp
export CC := gcc
export LD := gcc
export LIB := ar
export RAN := ranlib
export DOXYGEN := doxygen

# Still to define down here - BUILDDIR, CCflags ##


# Local variables:

TARGETS = $(wildcard SimCommsys/*)

# Master targets

all:     debug release

profile:
	@$(MAKE) RELEASE=Profile DOTARGET=all $(TARGETS)

release:
	@$(MAKE) RELEASE=Release DOTARGET=all $(TARGETS)

debug:
	@$(MAKE) RELEASE=Debug DOTARGET=all $(TARGETS)

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

showsettings:
	$(CC) $(CCflagRelease) -Q --help=target --help=optimizers --help=warnings

FORCE:

.PHONY:	all debug release profile install clean

# Manual targets

$(TARGETS):	$(LIBS) FORCE
	@echo "----> Making target \"$(notdir $@)\" [$(RELEASE)]."
	@$(MAKE) -C "$(ROOTDIR)/$@" $(DOTARGET)

# Pattern-matched targets

%.a:	FORCE
	@echo "----> Making library \"$(notdir $@)\" [$(RELEASE)]."
	@$(MAKE) -C $(ROOTDIR)/Libraries/$(patsubst lib%.a,Lib%,$(notdir $@)) $(DOTARGET)
