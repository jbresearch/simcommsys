#include "vcs.h"

#include <iostream>

namespace libbase {

const vcs vcs::version("Version Control System module (vcs)", 1.10);

vcs::vcs(const char *name, const double version, const char *build)
   {           
   using std::ios;
   using std::clog;

   const ios::fmtflags flags = clog.flags();
   clog.setf(ios::fixed, ios::floatfield);
   int prec = clog.precision(2);
   clog << "# VCS: " << name << " Version " << version << " (Build " << build << ")\n" << std::flush;
   clog.precision(prec);
   clog.flags(flags);
   }

}; // end namespace
