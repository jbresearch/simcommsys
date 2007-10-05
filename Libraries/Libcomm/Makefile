# All compiling, linking, and library flags are imported

LIBRARY := $(BUILDDIR)/libcomm.a

SOURCES := $(wildcard Source/*.cpp)
OBJECTS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.d)

# Master targets

.PHONY:	clean clean-dist clean-depend

all:	$(BUILDDIR) $(LIBRARY)

clean:
	$(RM) $(BUILDDIR)

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

$(LIBRARY):	$(OBJECTS)
	@echo "Building library $*."
	@$(LIB) $(LIBflags) $(LIBRARY) $(OBJECTS)
	@$(RAN) $(LIBRARY)

# Pattern-matched targets

$(BUILDDIR)/%.o:	Source/%.cpp
	@echo "Compiling $<."
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	Source/%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Making dependancy list for $*.o"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
