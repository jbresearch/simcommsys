/*!
 * \file
 * 
 * Copyright (c) 2010 Johann A. Briffa
 * 
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * \section svn Version Control
 * - $Id$
 */

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "rectangular.h"
#include <cstdlib>
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
            << "FATAL ERROR (rectangular): Interleaver block size cannot be greater than BCJR block." << std::endl;
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
   sout << this->lut.size() << std::endl;
   sout << rows << std::endl;
   sout << cols << std::endl;
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
