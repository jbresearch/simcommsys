/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "lapgauss.h"

namespace libcomm {

const libbase::vcs lapgauss::version("Additive Laplacian-Gaussian Channel module (lapgauss)", 1.22);

const libbase::serializer lapgauss::shelper("channel", "lapgauss", lapgauss::create);


// constructors / destructors

lapgauss::lapgauss()
   {
   }

// handle functions

void lapgauss::compute_parameters(const double Eb, const double No)
   {
   sigma = sqrt(Eb*No);
   }
   
// channel handle functions

sigspace lapgauss::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

double lapgauss::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   using libbase::gauss;
   sigspace n = rx - tx;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

// description output

std::string lapgauss::description() const
   {
   return "Laplacian-Gaussian channel";
   }

// object serialization - saving

std::ostream& lapgauss::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& lapgauss::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
