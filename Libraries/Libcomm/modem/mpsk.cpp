/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "mpsk.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

const libbase::serializer mpsk::shelper("blockmodem", "mpsk", mpsk::create);

// initialization

void mpsk::init(const int m)
   {
   lut.init(m);
   // allocate symbols using a Gray code sequence
   using libbase::gray;
   for (int i = 0; i < m; i++)
      {
      const double r = 1;
      const double theta = i * (2 * libbase::PI / m);
      lut(gray(i)) = sigspace(r * cos(theta), r * sin(theta));
      }
   }

// description output

std::string mpsk::description() const
   {
   std::ostringstream sout;
   switch (lut.size())
      {
      case 2:
         sout << "BPSK blockmodem";
         break;
      case 4:
         sout << "Gray QPSK blockmodem";
         break;
      default:
         sout << "Gray " << lut.size() << "PSK blockmodem";
         break;
      }
   return sout.str();
   }

// object serialization - saving

std::ostream& mpsk::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   return sout;
   }

// object serialization - loading

std::istream& mpsk::serialize(std::istream& sin)
   {
   int m;
   sin >> libbase::eatcomments >> m;
   init(m);
   return sin;
   }

} // end namespace
