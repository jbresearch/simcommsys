# All compiling, linking, and library flags are imported

SOURCES := $(wildcard Source/*.cpp)
OBJECTS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.d)
TARGETS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%)
FINAL := $(SOURCES:Source/%.cpp=$(BINDIR)/%.$(RELEASE))

# Master targets

.PHONY:	all install clean

all:	$(TARGETS) $(BUILDDIR)

install: $(FINAL) $(BINDIR)

clean:
	$(RM) $(BUILDDIR)

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

$(BINDIR):
	@$(MKDIR) $(BINDIR)

# Pattern-matched targets

$(BINDIR)/%.$(RELEASE):	$(BUILDDIR)/% $(BINDIR)
	@echo "Installing $* [$(RELEASE)]"
	@$(CP) $< $@

$(BUILDDIR)/%:	$(BUILDDIR)/%.o $(BUILDDIR) $(LIBS)
	@echo "Linking $* [$(RELEASE)]"
	@$(LD) $(LDflags) -o $@ $< $(LDlibs)

$(BUILDDIR)/%.o:	Source/%.cpp $(BUILDDIR)
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	Source/%.cpp $(BUILDDIR)
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
