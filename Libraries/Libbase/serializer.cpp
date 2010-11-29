/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "serializer.h"
#include <iostream>

namespace libbase {

using std::hex;
using std::dec;

// Determine debug level:
// 1 - Normal debug output only
// 2 - Trace calls
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// static variables

std::map<std::string, serializer::fptr>* serializer::cmap = NULL;
int serializer::count = 0;

// static functions

serializable* serializer::call(const std::string& base, const std::string& derived)
   {
   fptr func = (*cmap)[base + ":" + derived];
#if DEBUG>=2
   trace << "DEBUG (serializer): call(" << base+":"+derived << ") = " << (void *)func << "." << std::endl;
#endif
   if (func == NULL)
      return NULL;
   return (*func)();
   }

// constructor / destructor

serializer::serializer(const std::string& base, const std::string& derived,
      fptr func)
   {
   if (cmap == NULL)
      cmap = new std::map<std::string, fptr>;
#if DEBUG>=2
   trace << "DEBUG (serializer): new map entry [" << count << "] for (" << base+":"+derived << ") = " << (void *)func << "." << std::endl;
#endif
   (*cmap)[base + ":" + derived] = func;
   classname = derived;
   count++;
   }

serializer::~serializer()
   {
   count--;
   if (count == 0)
      delete cmap;
   }

} // end namespace
