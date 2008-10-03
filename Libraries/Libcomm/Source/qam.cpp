/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "qam.h"
#include "itfunc.h"
#include <math.h>
#include <sstream>

namespace libcomm {

const libbase::serializer qam::shelper("blockmodem", "qam", qam::create);

// Internal operations

/*! \brief Initialization

   For square constellations, allocate symbols using Gray code sequences for
   rows and columns (i.e adjacent symbols in the quadrature and in-phase directions
   respectively); each symbol then represents the concatenation of these two Gray
   code values.

   The constellation is centered about the origin, and uses the conventional
   scaling where symbols take amplitudes +/- 1, 3, 5, ...
   
   \sa Sklar, 2nd edition, p.565.
   
   \sa Matlab bin2gray function from the communications toolbox
*/
void qam::init(const int m)
   {
   using libbase::gray;
   const int k = int(log2(m));  // number of bits per symbol
   if(m != 1<<k)
      {
      std::cerr << "FATAL ERROR (qam): non-binary constellations not supported (" << m << ")\n";
      exit(1);
      }
   if(k % 1) // non-square constellation
      {
      std::cerr << "FATAL ERROR (qam): non-square constellations not supported (" << m << ")\n";
      exit(1);
      }
   // preliminaries
   lut.init(m);
   const int s = 1 << k/2;
   // start by setting up symbols in the first quadrant
   for(int i=0; i<s; i++)
      for(int q=0; q<s; q++)
         {
         const int c = (gray(i) << k/2) + gray(q);
         lut(c) = sigspace(i,q) * 2.0;
         }
   // translate, so that constellation is symmetric about the origin
   lut -= sigspace(s-1,s-1);
   }

// description output

std::string qam::description() const
   {
   std::ostringstream sout;
   sout << "Gray " << lut.size() << "QAM blockmodem";
   return sout.str();
   }

// object serialization - saving

std::ostream& qam::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   return sout;
   }

// object serialization - loading

std::istream& qam::serialize(std::istream& sin)
   {
   int m;
   sin >> m;
   init(m);
   return sin;
   }

}; // end namespace
