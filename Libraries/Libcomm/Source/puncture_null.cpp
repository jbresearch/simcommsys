#include "puncture_null.h"

namespace libcomm {

const libbase::vcs puncture_null::version("Null Puncturing System module (puncture_null)", 2.30);

const libbase::serializer puncture_null::shelper("puncture", "null", puncture_null::create);


// initialization

void puncture_null::init(const int tau)
   {
   // initialise the pattern matrix
   libbase::matrix<bool> pattern(tau,1);
   pattern = 1;
   // fill-in remaining variables
   puncture::init(pattern);
   }

// description output

std::string puncture_null::description() const
   {
   return "Unpunctured";
   }

// object serialization - saving

std::ostream& puncture_null::serialize(std::ostream& sout) const
   {
   sout << get_inputs() << "\n";
   return sout;
   }

// object serialization - loading

std::istream& puncture_null::serialize(std::istream& sin)
   {
   int tau;
   sin >> tau;
   init(tau);
   return sin;
   }

}; // end namespace
