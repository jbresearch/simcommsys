#include "shift_lut.h"
#include <sstream>

namespace libcomm {

const libbase::vcs shift_lut::version("Barrel-Shifting LUT Interleaver module (shift_lut)", 1.40);

const libbase::serializer shift_lut::shelper("interleaver", "shift", shift_lut::create);


// initialisation functions

void shift_lut::init(const int amount, const int tau)
   {
   shift_lut::amount = amount;

   lut.init(tau);
   for(int i=0; i<tau; i++)
      lut(i) = (i + amount) % tau;
   }

// description output

std::string shift_lut::description() const
   {
   std::ostringstream sout;
   sout << "Shift by " << amount << " Interleaver";
   return sout.str();
   }

// object serialization - saving

std::ostream& shift_lut::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   sout << amount << "\n";
   return sout;
   }

// object serialization - loading

std::istream& shift_lut::serialize(std::istream& sin)
   {
   int tau, amount;
   sin >> tau >> amount;
   init(amount, tau);
   return sin;
   }

}; // end namespace
