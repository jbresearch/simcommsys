#include "vcs.h"
#include <iostream.h>

const vcs vcs_version("Version Control System module (vcs)", 1.00);

vcs::vcs(const char *name, const double version, const char *build)
   {           
   int flags = cout.flags();
   cout.setf(ios::fixed, ios::floatfield);
   int prec = cout.precision(2);
   cout << "# VCS: " << name << " Version " << version << " (Build " << build << ")\n";
   cout.precision(prec);
   cout.flags(flags);
   }
