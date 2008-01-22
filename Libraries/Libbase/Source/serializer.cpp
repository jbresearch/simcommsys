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

using std::hex;
using std::dec;
using std::flush;

// static variables

std::map<std::string,serializer::fptr>* serializer::cmap = NULL;
int serializer::count = 0;

// static functions

void* serializer::call(const std::string& base, const std::string& derived)
   {
   fptr func = (*cmap)[base+":"+derived];
   trace << "DEBUG (serializer): call(" << base+":"+derived << ") = 0x" << hex << (void *)func << dec << ".\n";
   if(func == NULL)
      return NULL;
   return (*func)();
   }

// constructor / destructor

serializer::serializer(const std::string& base, const std::string& derived, fptr func)
   {
   if(cmap == NULL)
      cmap = new std::map<std::string,fptr>;
   trace << "DEBUG (serializer): new map entry [" << count << "] for (" << base+":"+derived << ") = 0x" << hex << (void *)func << dec << ".\n";
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
