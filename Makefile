# Exported variables:

# Directory where the binaries are placed
OSNAME := $(shell uname -s)-$(shell uname -i)
export BUILDDIR := Arch/$(OSNAME)

# Linker settings
LDlibusr := -lcomm -lbase
LDlibsys := -lm -lstdc++
#LDlibmpi := `mpic++ -showme:link`
#LDlibmpi := -lpmpich++ -lmpich
LDlibmpi :=
export LDlibs := $(LDlibusr) $(LDlibsys) $(LDlibmpi)
export LDflags := $(LDlibusr:-l%=-L$(PWD)/Libraries/Lib%/$(BUILDDIR))

# Compiler settings
CCoptions := -O3 -DNDEBUG
CClibs := $(LDlibusr:-l%=-I$(PWD)/Libraries/Lib%/Source)
CClang := -Wall -Wno-non-template-friend
#CCmpi := -DUSEMPI `mpic++ -showme:compile`
#CCmpi := -DUSEMPI -DUSE_STDARG -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STDARG_H=1 -DUSE_STDARG=1 -DMALLOC_RET_VOID=1
CCmpi :=
# Define compiling flags
export CCflags := $(CCoptions) $(CClibs) $(CClang) $(CCmpi)
export CCdepend := -MM

# User library list
export LIBS := $(foreach name,$(LDlibusr:-l%=%),$(PWD)/Libraries/Lib$(name)/$(BUILDDIR)/lib$(name).a)

# Library builder settings
export LIBflags := ru

# Define the names for commands
export MKDIR := mkdir -p
export RM := rm -rf
export CC := gcc
export LD := gcc
export LIB := ar
export RAN := ranlib


# Local variables:

TARGETS := Turbo\ Codes

# Master targets

.PHONY:	clean clean-dist clean-depend

all:	$(TARGETS)

FORCE:

# Manual targets

$(TARGETS):	$(LIBS) FORCE
	@echo "----> Making target \"$(notdir $@)\"."
	@$(MAKE) -C "$(PWD)/$@"

# Pattern-matched targets

%.a:	FORCE
	@echo "----> Making library \"$(notdir $@)\"."
	@$(MAKE) -C $(PWD)/Libraries/$(patsubst lib%.a,Lib%,$(notdir $@))
