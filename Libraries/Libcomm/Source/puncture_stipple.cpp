/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "puncture_stipple.h"
#include <sstream>

namespace libcomm {

const libbase::vcs puncture_stipple::version("Stippled Puncturing System module (puncture_stipple)", 2.31);

const libbase::serializer puncture_stipple::shelper("puncture", "stipple", puncture_stipple::create);


// initialization

void puncture_stipple::init(const int tau, const int sets)
   {
   puncture_stipple::tau = tau;
   puncture_stipple::sets = sets;
   // initialise the pattern matrix
   libbase::matrix<bool> pattern(tau,sets);
   for(int t=0; t<tau; t++)
      for(int s=0; s<sets; s++)
         pattern(t,s) = (s==0 || (s-1)==t%(sets-1));
   // fill-in remaining variables
   puncture::init(pattern);
   }

// description output

std::string puncture_stipple::description() const
   {
   std::ostringstream sout;
   sout << "Stipple Puncturing (" << tau << "x" << sets << ", " << get_outputs() << "/" << get_inputs() << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& puncture_stipple::serialize(std::ostream& sout) const
   {
   sout << tau << "\n";
   sout << sets << "\n";
   return sout;
   }

// object serialization - loading

std::istream& puncture_stipple::serialize(std::istream& sin)
   {
   sin >> tau >> sets;
   init(tau, sets);
   return sin;
   }

}; // end namespace
