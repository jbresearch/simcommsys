/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "helical.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// initialisation functions

template <class real>
void helical<real>::init(const int tau, const int rows, const int cols)
   {
   helical<real>::rows = rows;
   helical<real>::cols = cols;

   int blklen = rows*cols;
   if(blklen > tau)
      {
      std::cerr << "FATAL ERROR (helical): Interleaver block size cannot be greater than BCJR block.\n";
      exit(1);
      }
   this->lut.init(tau);
   int row = rows-1, col = 0;
   int i;
   for(i=0; i<blklen; i++)
      {
      this->lut(i) = row*cols + col;
      row = (row-1+rows) % rows;
      col = (col+1) % cols;
      }
   for(i=blklen; i<tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string helical<real>::description() const
   {
   std::ostringstream sout;
   sout << "Helical " << rows << "x" << cols << " Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& helical<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << "\n";
   sout << rows << "\n";
   sout << cols << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& helical<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> tau;
   sin >> rows;
   sin >> cols;
   init(tau, rows, cols);
   return sin;
   }

// Explicit instantiations

template class helical<float>;
template <>
const libbase::serializer helical<float>::shelper("interleaver", "helical<float>", helical<float>::create);

template class helical<double>;
template <>
const libbase::serializer helical<double>::shelper("interleaver", "helical<double>", helical<double>::create);

template class helical<libbase::logrealfast>;
template <>
const libbase::serializer helical<libbase::logrealfast>::shelper("interleaver", "helical<logrealfast>", helical<libbase::logrealfast>::create);

}; // end namespace
