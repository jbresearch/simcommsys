#include "berrou.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

const libbase::vcs berrou::version("Berrou's Original Interleaver module (berrou)", 1.41);

const libbase::serializer berrou::shelper("interleaver", "berrou", berrou::create);

// initialization

void berrou::init(const int M)
   {
   berrou::M = M;

   if(libbase::weight(M) != 1)
      {
      std::cerr << "FATAL ERROR (berrou): M must be an integral power of 2.\n";
      exit(1);
      }
   int tau = M*M;
   lut.init(tau);
   const int P[] = {17, 37, 19, 29, 41, 23, 13, 7};
   for(int i=0; i<M; i++)
      for(int j=0; j<M; j++)
         {
         int ir = (M/2+1)*(i+j) % M;
         int xi = (i+j) % 8;
         int jr = (P[xi]*(j+1) - 1) % M;
         lut(i*M+j) = ir*M + jr;
         }
   }

// description output

std::string berrou::description() const
   {
   std::ostringstream sout;
   sout << "Berrou Interleaver (" << M << "x" << M << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& berrou::serialize(std::ostream& sout) const
   {
   sout << M << "\n";
   return sout;
   }

// object serialization - loading

std::istream& berrou::serialize(std::istream& sin)
   {
   sin >> M;
   init(M);
   return sin;
   }

}; // end namespace
