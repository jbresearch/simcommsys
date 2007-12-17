/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "serializer.h"
#include <iostream>

//#define DEBUG

namespace libbase {

using std::clog;
using std::hex;
using std::dec;
using std::flush;

const vcs serializer::version("Serialization helper module (serializer)", 1.20);

// static variables

std::map<std::string,serializer::fptr>* serializer::cmap = NULL;
int serializer::count = 0;

// static functions

void* serializer::call(const std::string& base, const std::string& derived)
   {
   fptr func = (*cmap)[base+":"+derived];
#ifdef DEBUG
   clog << "DEBUG (serializer): call(" << base+":"+derived << ") = 0x" << hex << func << dec << ".\n";
#endif
   if(func == NULL)
      return NULL;
   return (*func)();
   }

// constructor / destructor

serializer::serializer(const std::string& base, const std::string& derived, fptr func)
   {
   if(cmap == NULL)
      cmap = new std::map<std::string,fptr>;
#ifdef DEBUG
   clog << "DEBUG (serializer): new map entry [" << count << "] for (" << base+":"+derived << ") = 0x" << hex << func << dec << ".\n";
#endif
   (*cmap)[base+":"+derived] = func;
   classname = derived;
   count++;
   }

serializer::~serializer()
   {
   count--;
   if(count == 0)
      delete cmap;
   }

}; // end namespace
