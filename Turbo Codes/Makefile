# All compiling, linking, and library flags are imported

SOURCES := $(wildcard Source/*.cpp)
OBJECTS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.d)
TARGETS := $(SOURCES:Source/%.cpp=$(BINDIR)/%.$(RELEASE))

# Master targets

.PHONY:	clean clean-dist clean-depend

all:	$(BUILDDIR) $(BINDIR) $(TARGETS)

clean:
	$(RM) $(BUILDDIR)
	$(RM) $(BINDIR)/*.$(RELEASE)

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

$(BINDIR):
	@$(MKDIR) $(BINDIR)

# Pattern-matched targets

$(BINDIR)/%.$(RELEASE):	$(BUILDDIR)/%.o $(LIBS)
	@echo "Linking $* [$(RELEASE)]"
	@$(LD) $(LDflags) -o $@ $< $(LDlibs)

$(BUILDDIR)/%.o:	Source/%.cpp
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	Source/%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
