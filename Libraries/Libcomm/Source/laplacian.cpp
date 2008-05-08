/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "laplacian.h"

namespace libcomm {

const libbase::serializer laplacian::shelper("channel", "laplacian", laplacian::create);

// handle functions

void laplacian::compute_parameters(const double Eb, const double No)
   {
   const double sigma = sqrt(Eb*No);
   lambda = sigma/sqrt(double(2));
   }

// channel handle functions

sigspace laplacian::corrupt(const sigspace& s)
   {
   const double x = Finv(r.fval());
   const double y = Finv(r.fval());
   return s + sigspace(x, y);
   }

double laplacian::pdf(const sigspace& tx, const sigspace& rx) const
   {
   sigspace n = rx - tx;
   return f(n.i()) * f(n.q());
   }

// description output

std::string laplacian::description() const
   {
   return "Laplacian channel";
   }

// object serialization - saving

std::ostream& laplacian::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& laplacian::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
