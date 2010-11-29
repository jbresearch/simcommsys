/*!
 * \file
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "assertalways.h"

#include <iostream>
#include <string>
#include <cstdlib>

namespace libbase {

// Debugging tools

void reportassertionandfail(const char *expression, const char *file, int line)
   {
   std::string s;
   s = "assertion " + std::string(expression) + " failed.";
   reporterrorandfail(s.c_str(), file, line);
   }

void reporterrorandfail(const char *expression, const char *file, int line)
   {
   std::cerr << "ERROR (" << file << " line " << line << "): " << expression
         << std::endl;
   exit(1);
   }

} // end namespace
