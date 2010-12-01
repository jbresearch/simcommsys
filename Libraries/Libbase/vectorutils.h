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

#ifndef VECTORUTILS_H_
#define VECTORUTILS_H_

#include "vector.h"
#include "matrix.h"

namespace libbase {

/*! \brief Allocates memory for a vector of vectors
 * For object vv, allocates memory such that vv has 'outer' vectors each of
 * size 'inner'.
 */
template <class T>
void allocate(vector<vector<T> >& vv, const int outer, const int inner)
   {
   vv.init(outer);
   for (int i = 0; i < outer; i++)
      vv(i).init(inner);
   }

/*! \brief Allocates memory for a vector of matrices
 * For object vm, allocates memory such that vm has 'outer' matrix each of
 * size 'innerrows' by 'innercols'.
 */
template <class T>
void allocate(vector<matrix<T> >& vm, const int outer, const int innerrows,
      const int innercols)
   {
   vm.init(outer);
   for (int i = 0; i < outer; i++)
      vm(i).init(innerrows, innercols);
   }

/*! \brief Allocates memory for a matrix of vectors
 * For object mv, allocates memory such that mv has 'outerrows' by 'outercols'
 * vectors each of size 'inner'.
 */
template <class T>
void allocate(matrix<vector<T> >& mv, const int outerrows, const int outercols,
      const int inner)
   {
   mv.init(outerrows, outercols);
   for (int i = 0; i < outerrows; i++)
      for (int j = 0; j < outercols; j++)
         mv(i, j).init(inner);
   }

} // end namespace

#endif /* VECTORUTILS_H_ */
