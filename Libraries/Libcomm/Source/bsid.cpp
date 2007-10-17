#include "bsid.h"

namespace libcomm {

const libbase::vcs bsid::version("Binary Substitution, Insertion, and Deletion Channel module (bsid)", 1.01);

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);


// constructors / destructors

bsid::bsid()
   {
   }

// handle functions

void bsid::compute_parameters(const double Eb, const double No)
   {
   Ps = 0.5*erfc(1/sqrt(Eb*No*2));
   }
   
// channel handle functions

sigspace bsid::corrupt(const sigspace& s)
   {
   const double p = r.fval();
   if(p < Ps)
      return -s;
   return s;
   }

double bsid::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   if(tx != rx)
      return Ps;
   return 1-Ps;
   }

// channel functions
   
// description output

std::string bsid::description() const
   {
   return "BSID channel";
   }

// object serialization - saving

std::ostream& bsid::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& bsid::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
