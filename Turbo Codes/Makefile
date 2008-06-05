# All compiling, linking, and library flags are imported

SOURCES := $(wildcard Source/*.cpp)
OBJECTS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%.d)
TARGETS := $(SOURCES:Source/%.cpp=$(BUILDDIR)/%)
FINAL := $(SOURCES:Source/%.cpp=$(BINDIR)/%.$(WCTAG).$(RELEASE))

# Master targets

all:	$(TARGETS) $(BUILDDIR)

install: $(FINAL) $(BINDIR)

clean:
	$(RM) $(BUILDDIR)

.PHONY:	all install clean

# Manual targets

$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)

$(BINDIR):
	@$(MKDIR) $(BINDIR)

# Pattern-matched targets

$(BINDIR)/%.$(WCTAG).$(RELEASE):	$(BUILDDIR)/%
	@echo "Installing $* [$(RELEASE)]"
	@$(CP) $< $@

$(BUILDDIR)/%:	$(BUILDDIR)/%.o $(LIBS)
	@echo "Linking $* [$(RELEASE)]"
	@$(LD) $(LDflags) -o $@ $< $(LDlibs)

$(BUILDDIR)/%.o:	Source/%.cpp
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	Source/%.cpp
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(MKDIR) $(BUILDDIR)
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

include $(DEPEND)
