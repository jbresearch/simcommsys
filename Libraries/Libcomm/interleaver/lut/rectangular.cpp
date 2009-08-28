/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "rectangular.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// initialisation functions

template <class real>
void rectangular<real>::init(const int tau, const int rows, const int cols)
   {
   rectangular<real>::rows = rows;
   rectangular<real>::cols = cols;

   int blklen = rows * cols;
   if (blklen > tau)
      {
      std::cerr
            << "FATAL ERROR (rectangular): Interleaver block size cannot be greater than BCJR block.\n";
      exit(1);
      }
   this->lut.init(tau);
   int row = 0, col = 0;
   int i;
   for (i = 0; i < blklen; i++)
      {
      row = i % rows;
      col = i / rows;
      this->lut(i) = row * cols + col;
      }
   for (i = blklen; i < tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string rectangular<real>::description() const
   {
   std::ostringstream sout;
   sout << "Rectangular " << rows << "x" << cols << " Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& rectangular<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << "\n";
   sout << rows << "\n";
   sout << cols << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& rectangular<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> libbase::eatcomments >> tau;
   sin >> libbase::eatcomments >> rows;
   sin >> libbase::eatcomments >> cols;
   init(tau, rows, cols);
   return sin;
   }

// Explicit instantiations

template class rectangular<float> ;
template <>
const libbase::serializer rectangular<float>::shelper("interleaver",
      "rectangular<float>", rectangular<float>::create);

template class rectangular<double> ;
template <>
const libbase::serializer rectangular<double>::shelper("interleaver",
      "rectangular<double>", rectangular<double>::create);

template class rectangular<libbase::logrealfast> ;
template <>
const libbase::serializer rectangular<libbase::logrealfast>::shelper(
      "interleaver", "rectangular<logrealfast>", rectangular<
            libbase::logrealfast>::create);

} // end namespace
