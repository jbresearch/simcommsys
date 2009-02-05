/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "rand_lut.h"
#include "vector.h"
#include <sstream>

namespace libcomm {

// initialisation

template <class real>
void rand_lut<real>::init(const int tau, const int m)
   {
   p = (1<<m)-1;
   if(tau % p != 0)
      {
      std::cerr << "FATAL ERROR (rand_lut): interleaver length must be a multiple of the encoder impulse respone length.\n";
      exit(1);
      }
   this->lut.init(tau);
   }

// intra-frame functions

template <class real>
void rand_lut<real>::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

template <class real>
void rand_lut<real>::advance()
   {
   const int tau = this->lut.size();
   // create array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau);
   used = false;
   // fill in lut
   for(int t=0; t<tau; t++)
      {
      int tdash;
      do {
         tdash = int(r.ival(tau)/p)*p + t%p;
         } while(used(tdash));
      used(tdash) = true;
      this->lut(t) = tdash;
      }
   }

// description output

template <class real>
std::string rand_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Random Interleaver (self-terminating for m=" << int(log2(p+1)) << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& rand_lut<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << "\n";
   sout << int(log2(p+1)) << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& rand_lut<real>::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> tau >> m;
   init(tau, m);
   return sin;
   }

// Explicit instantiations

template class rand_lut<double>;
template <>
const libbase::serializer rand_lut<double>::shelper("interleaver", "rand_lut<double>", rand_lut<double>::create);

template class rand_lut<libbase::logrealfast>;
template <>
const libbase::serializer rand_lut<libbase::logrealfast>::shelper("interleaver", "rand_lut<logrealfast>", rand_lut<libbase::logrealfast>::create);

}; // end namespace
