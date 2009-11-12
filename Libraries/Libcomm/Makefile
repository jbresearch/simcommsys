# $Author: jabriffa $
# $Revision: 1947 $
# $Date: 2008-11-05 11:22:31 +0000 (Wed, 05 Nov 2008) $

# All compiling, linking, and library flags are imported

SOURCES := $(shell find . -name '*.cpp')
OBJECTS := $(SOURCES:%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:%.cpp=$(BUILDDIR)/%.d)
LIBRARY := $(BUILDDIR)/libcomm.a

# Master targets

all:	$(LIBRARY)

install:	all

clean:
	@echo "Cleaning [$(RELEASE)]"
	@$(RM) $(BUILDDIR)

.PHONY:	all install clean

# Manual targets

$(LIBRARY):	$(OBJECTS)
	@echo "Building library $* [$(RELEASE)]"
	@$(LIB) $(LIBflags) $(LIBRARY) $(OBJECTS)
	@$(RAN) $(LIBRARY)

# Pattern-matched targets

$(BUILDDIR)/%.o:	%.cpp
	@$(MKDIR) $(dir $@)
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	%.cpp
	@$(MKDIR) $(dir $@)
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(CC) $(CCflags) $(CCdepend) -MF$@ $< 

ifneq ($(MAKECMDGOALS),clean)
include $(DEPEND)
endif
