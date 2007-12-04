#include "vale96int.h"

namespace libcomm {

const libbase::vcs vale96int::version("Matt Valenti's Interleaver module (vale96int)", 1.30);


// initialisation functions

vale96int::vale96int()
   {
   // set name and forced tail length
   lutname = "vale96int";
   m = 0;
   // build LUT
   const int tau = 34;
   const int a[] = {16, 29, 9, 10, 14, 6, 31, 8, 12, 22, 17, 33, 34, 23, 24, 19, 32, 30, 13, 2, 21, 25, 26, 3, 28, 20, 27, 7, 5, 15, 4, 11, 18, 1};
   lut.init(tau);
   lut = -1;
   for(int i=0; i<tau; i++)
      {
      const int ndx = a[i]-1;
      if(lut(ndx) != -1)
         {
         std::cerr << "DEBUG ERROR (vale96int): Duplicate entry (" << a[i] << ") at position " << i << " (previous one at " << lut(ndx) << ")\n";
         exit(1);
         }
      lut(ndx) = i;
      }
   }

}; // end namespace
