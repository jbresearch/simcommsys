#include "laplacian.h"

namespace libcomm {

const libbase::vcs laplacian::version("Additive Laplacian Noise Channel module (laplacian)", 1.43);

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

}; // end namespace
