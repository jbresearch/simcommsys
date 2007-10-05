#ifndef __vcs_h
#define __vcs_h

class vcs {
public:
   vcs(const char *name, const double version, const char *build = __DATE__);
};

extern const vcs vcs_version;

#endif

