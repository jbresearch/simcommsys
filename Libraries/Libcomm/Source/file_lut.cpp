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

// creation/destruction functions

template <class real>
file_lut<real>::file_lut(const char *filename, const int tau, const int m)
   {
   file_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s==NULL) ? filename : s+1;
   this->lutname = p;

   this->lut.init(tau);

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
      this->lut(i) = y;
      }
   for(int t=tau-m; t<tau; t++)
      this->lut(t) = fsm::tail;
   fclose(file);
   }

// Explicit instantiations

template class file_lut<double>;
template class file_lut<libbase::logrealfast>;

}; // end namespace
