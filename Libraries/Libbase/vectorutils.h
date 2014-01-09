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
 */

#ifndef VECTORUTILS_H_
#define VECTORUTILS_H_

#include "vector.h"
#include "matrix.h"

/*!
 * \file
 * \brief   Generic Utility Functions for Vectors.
 * \author  Johann Briffa
 *
 */

namespace libbase {

/*! \brief Returns the index of the min value
 * \param v vector to examine
 * \param getfirst flag to return first value found (rather than last)
 */
template <class T>
int index_of_min(const vector<T>& v, const bool getfirst = true)
   {
   int index;
   v.min(index, getfirst);
   return index;
   }

/*! \brief Returns the index of the max value
 * \param v vector to examine
 * \param getfirst flag to return first value found (rather than last)
 */
template <class T>
int index_of_max(const vector<T>& v, const bool getfirst = true)
   {
   int index;
   v.max(index, getfirst);
   return index;
   }

/*! \brief Concatenates vectors
 * Creates a vector containing a concatenation of two vectors.
 */
template <class T>
vector<T> concatenate(const vector<T>& v1, const vector<T>& v2)
   {
   // shorthand for sizes
   const int s1 = v1.size();
   const int s2 = v2.size();
   // allocate memory for result
   vector<T> result;
   result.init(s1 + s2);
   // do the concatenation
   result.segment(0, s1) = v1;
   result.segment(s1, s2) = v2;
   return result;
   }

/*! \brief Concatenates vectors
 * Creates a vector containing a concatenation of three vectors.
 */
template <class T>
vector<T> concatenate(const vector<T>& v1, const vector<T>& v2,
      const vector<T>& v3)
   {
   // shorthand for sizes
   const int s1 = v1.size();
   const int s2 = v2.size();
   const int s3 = v3.size();
   // allocate memory for result
   vector<T> result;
   result.init(s1 + s2 + s3);
   // do the concatenation
   result.segment(0, s1) = v1;
   result.segment(s1, s2) = v2;
   result.segment(s1 + s2, s3) = v3;
   return result;
   }

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
