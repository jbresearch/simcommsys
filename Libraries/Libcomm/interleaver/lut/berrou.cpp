/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "berrou.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// initialization

template <class real>
void berrou<real>::init(const int M)
   {
   berrou<real>::M = M;

   if (libbase::weight(M) != 1)
      {
      std::cerr << "FATAL ERROR (berrou): M must be an integral power of 2." << std::endl;
      exit(1);
      }
   int tau = M * M;
   this->lut.init(tau);
   const int P[] = {17, 37, 19, 29, 41, 23, 13, 7};
   for (int i = 0; i < M; i++)
      for (int j = 0; j < M; j++)
         {
         int ir = (M / 2 + 1) * (i + j) % M;
         int xi = (i + j) % 8;
         int jr = (P[xi] * (j + 1) - 1) % M;
         this->lut(i * M + j) = ir * M + jr;
         }
   }

// description output

template <class real>
std::string berrou<real>::description() const
   {
   std::ostringstream sout;
   sout << "Berrou Interleaver (" << M << "x" << M << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& berrou<real>::serialize(std::ostream& sout) const
   {
   sout << M << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& berrou<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> M;
   init(M);
   return sin;
   }

// Explicit instantiations

template class berrou<float> ;
template <>
const libbase::serializer berrou<float>::shelper("interleaver",
      "berrou<float>", berrou<float>::create);

template class berrou<double> ;
template <>
const libbase::serializer berrou<double>::shelper("interleaver",
      "berrou<double>", berrou<double>::create);

template class berrou<libbase::logrealfast> ;
template <>
const libbase::serializer berrou<libbase::logrealfast>::shelper("interleaver",
      "berrou<logrealfast>", berrou<libbase::logrealfast>::create);

} // end namespace
