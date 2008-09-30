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

template <class real>
stream_lut<real>::stream_lut(const char *filename, FILE *file, const int tau, const int m)
   {
   stream_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s==NULL) ? filename : s+1;
   this->lutname = p;

   this->lut.init(tau);

   char buf[256];
   for(int i=0; i<tau-m; i++)
      {
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(buf[0] == '#');
      int y;
      sscanf(buf, "%d", &y);
      this->lut(i) = y;
      }
   for(int t=tau-m; t<tau; t++)
      this->lut(t) = fsm::tail;
   }

// Explicit instantiations

template class stream_lut<double>;
template class stream_lut<libbase::logrealfast>;

}; // end namespace
