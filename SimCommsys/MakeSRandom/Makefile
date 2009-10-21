# All compiling, linking, and library flags are imported

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:%.cpp=$(BUILDDIR)/%.o)
DEPEND := $(SOURCES:%.cpp=$(BUILDDIR)/%.d)
TARGETS := $(SOURCES:%.cpp=$(BUILDDIR)/%)
FINAL := $(SOURCES:%.cpp=$(BINDIR)/%.$(WCTAG).$(RELEASE))

# Master targets

all:	$(BUILDDIR) $(TARGETS)

install: $(BINDIR) $(FINAL)

clean:
	@echo "Cleaning [$(RELEASE)]"
	@$(RM) $(BUILDDIR)

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

$(BUILDDIR)/%.o:	%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Compiling $< [$(RELEASE)]"
	@$(CC) $(CCflags) -c $< -o $@

$(BUILDDIR)/%.d:	%.cpp
	@$(MKDIR) $(BUILDDIR)
	@echo "Making dependancy list for $*.o [$(RELEASE)]"
	@$(CC) $(CCflags) $(CCdepend) $< | gawk 'BEGIN { FS = "$*.o" } { if( NF > 1 ) { print $$1, "$(BUILDDIR)/$*.d $(BUILDDIR)/$*.o", $$2 } else { print $$0 } }' > $@

#include $(DEPEND)
