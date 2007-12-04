#include "rand_lut.h"
#include "itfunc.h"
#include "vector.h"
#include <sstream>

namespace libcomm {

const libbase::vcs rand_lut::version("Random LUT Interleaver module (rand_lut)", 1.41);

const libbase::serializer rand_lut::shelper("interleaver", "random", rand_lut::create);


// initialisation

void rand_lut::init(const int tau, const int m)
   {
   p = (1<<m)-1;
   if(tau % p != 0)
      {
      std::cerr << "FATAL ERROR (rand_lut): interleaver length must be a multiple of the encoder impulse respone length.\n";
      exit(1);
      }
   lut.init(tau);
   seed(0);
   }

// intra-frame functions

void rand_lut::seed(const int s)
   {
   r.seed(s);
   advance();
   }

void rand_lut::advance()
   {
   const int tau = lut.size();
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
      lut(t) = tdash;
      }
   }

// description output

std::string rand_lut::description() const
   {
   std::ostringstream sout;
   sout << "Random Interleaver (self-terminating for m=" << int(libbase::log2(p+1)) << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& rand_lut::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   sout << int(libbase::log2(p+1)) << "\n";
   return sout;
   }

// object serialization - loading

std::istream& rand_lut::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> tau >> m;
   init(tau, m);
   return sin;
   }

}; // end namespace
