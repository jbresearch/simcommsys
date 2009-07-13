# $Author: jabriffa $
# $Revision: 1947 $
# $Date: 2008-11-05 11:22:31 +0000 (Wed, 05 Nov 2008) $

# Function definition for recursive wildcard

rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/)$(filter $(subst *,%,$2),$d))

# All compiling, linking, and library flags are imported

SOURCES := $(call rwildcard,,*.cpp)
OBJECTS := $(SOURCES:%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:%.cpp=$(BUILDDIR)/%.d)
LIBRARY := $(BUILDDIR)/libbase.a

# Master targets

.PHONY:	clean clean-dist clean-depend

all:	$(BUILDDIR) $(LIBRARY)

clean:
	$(RM) $(BUILDDIR)

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

$(LIBRARY):	$(OBJECTS)
	@echo "Building library $* [$(RELEASE)]"
	@$(LIB) $(LIBflags) $(LIBRARY) $(OBJECTS)
	@$(RAN) $(LIBRARY)

# Pattern-matched targets

$(BUILDDIR)/%.o:	%.cpp
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
