/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "stream_lut.h"

namespace libcomm {

// creation/destruction functions

stream_lut::stream_lut(const char *filename, FILE *file, const int tau, const int m)
   {
   stream_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s==NULL) ? filename : s+1;
   lutname = p;

   lut.init(tau);

   char buf[256];
   for(int i=0; i<tau-m; i++)
      {
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(buf[0] == '#');
      int x, y;
      sscanf(buf, "%d%d", &x, &y);
      if(x != i)
         {
         std::cerr << "FATAL ERROR (stream_lut): unexpected entry for line " << i << ": " << x << ", " << y << "\n";
         exit(1);
         }
      lut(i) = y;
      }
   for(int t=tau-m; t<tau; t++)
      lut(t) = tail;
   }

}; // end namespace
