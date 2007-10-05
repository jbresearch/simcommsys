#include "uniform_lut.h"
#include <sstream>

namespace libcomm {

const libbase::vcs uniform_lut::version("Uniform Interleaver module (uniform_lut)", 1.50);

const libbase::serializer uniform_lut::shelper("interleaver", "uniform", uniform_lut::create);


// initialisation

void uniform_lut::init(const int tau, const int m)
   {
   uniform_lut::tau = tau;
   uniform_lut::m = m;
   lut.init(tau);
   seed(0);
   }

// intra-frame functions

void uniform_lut::seed(const int s)
   {
   r.seed(s);
   advance();
   }

void uniform_lut::advance()
   {
   // create array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau-m);
   used = false;
   // fill in lut
   int t;
   for(t=0; t<tau-m; t++)
      {
      int tdash;
      do {
         tdash = r.ival(tau-m);
         } while(used(tdash));
      used(tdash) = true;
      lut(t) = tdash;
      }
   for(t=tau-m; t<tau; t++)
      lut(t) = tail;
   }

// description output

std::string uniform_lut::description() const
   {
   std::ostringstream sout;
   sout << "Uniform Interleaver";
   if(m > 0)
      sout << " (Forced tail length " << m << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& uniform_lut::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   sout << m << "\n";
   return sout;
   }

// object serialization - loading

std::istream& uniform_lut::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> tau >> m;
   init(tau, m);
   return sin;
   }

}; // end namespace
