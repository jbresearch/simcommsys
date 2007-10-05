# All compiling, linking, and library flags are imported

SOURCES := $(wildcard Source/*.cpp)
TARGETS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%)
OBJECTS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.d)

# Master targets

.PHONY:	clean clean-dist clean-depend

all:	$(BUILDDIR) $(TARGETS)

clean:
	$(RM) $(BUILDDIR)

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

# Pattern-matched targets

$(TARGETS):	%:	%.o $(LIBS)
	@echo "Linking $*."
	@$(LD) $(LDflags) -o $@ $< $(LDlibs)

$(BUILDDIR)/%.o:	Source/%.cpp
	@echo "Compiling $<."
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	Source/%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Making dependancy list for $*.o"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
