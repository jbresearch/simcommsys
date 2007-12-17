#ifndef __vcs_h
#define __vcs_h

#include "config.h"

namespace libbase {

/*
  Version 1.01 (29 Oct 2001)
  redirected the output to clog instead of cout

  Version 1.02 (23 Feb 2002)
  added flushes to all end-of-line clog outputs, to clean up text user interface.

  Version 1.03 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.04 (15 Jun 2002)
  changed 'flags' variable in implementation file from type int to type
  ios::fmtflags, as it's supposed to be.

  Version 1.10 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class vcs {
   static const vcs version;
public:
   vcs(const char *name, const double version, const char *build = __DATE__);
};

}; // end namespace

#endif

