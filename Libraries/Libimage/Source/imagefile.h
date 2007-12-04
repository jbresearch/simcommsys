#ifndef __imagefile_h
#define __imagefile_h

#include "config.h"
#include "vcs.h"

/*******************************************************************************
  
  Version 1.00 (7 Dec 2006)
  * initial version
  * class meant to encapsulate the data and functions dealing with image files,
  potentially containing several images.
  * this is effectively a container for images, together with the functions
  necessary to save and load to files or other streams.

*******************************************************************************/

namespace libimage {

class imagefile {
   static const libbase::vcs version;
public:
   // Construction / destruction
   imagefile();
   ~imagefile();

   // File format

   // Saving / loading
   
};

}; // end namespace

#endif
