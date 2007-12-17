/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "file_lut.h"
#include <stdio.h>

namespace libcomm {

const libbase::vcs file_lut::version("Pre-stored LUT Interleaver module (file_lut)", 1.40);


// creation/destruction functions

file_lut::file_lut(const char *filename, const int tau, const int m)
   {
   file_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s==NULL) ? filename : s+1;
   lutname = p;

   lut.init(tau);

   char buf[256];
   FILE *file = fopen(filename, "rb");
   if(file == NULL)
      {
      std::cerr << "FATAL ERROR (file_lut): Cannot open LUT file (" << filename << ").\n";
      exit(1);
      }
   for(int i=0; i<tau-m; i++)
      {
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(buf[0] == '#');
      int x, y;
      sscanf(buf, "%d%d", &x, &y);
      if(x != i)
         {
         std::cerr << "FATAL ERROR (file_lut): unexpected entry for line " << i << ": " << x << ", " << y << "\n";
         exit(1);
         }
      lut(i) = y;
      }
   for(int t=tau-m; t<tau; t++)
      lut(t) = tail;
   fclose(file);
   }

}; // end namespace
