#include "mpsk.h"
#include <math.h>
#include <sstream>

namespace libcomm {

const libbase::vcs mpsk::version("M-PSK Modulator module (mpsk)", 2.12);

const libbase::serializer mpsk::shelper("modulator", "mpsk", mpsk::create);


// initialization

void mpsk::init(const int m)
   {
   lut.init(m);
   // allocate symbols sequentially - this has to be changed
   // to use set-partitioning (for error-minimization)
   for(int i=0; i<m; i++)
      {
      const double r = 1;
      const double theta = i * (2*libbase::PI/m);
      lut(i) = sigspace(r*cos(theta), r*sin(theta));
      }
   }

// description output

std::string mpsk::description() const
   {
   std::ostringstream sout;
   switch(lut.size())
      {
      case 2:
         sout << "BPSK modulator";
         break;
      case 4:
         sout << "QPSK modulator";
         break;
      default:
         sout << lut.size() << "PSK modulator";
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
   sin >> m;
   init(m);
   return sin;
   }

}; // end namespace
