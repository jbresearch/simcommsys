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

#ifndef __hard_decision_h
#define __hard_decision_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "randgen.h"

#include <list>
#include <iostream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of tie-breaks
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 2
#endif

template <class dbl, class S>
class basic_hard_decision {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   /*! \name Internal object representation */
   libbase::randgen r; //!< Random source for resolving tie-breaks
#if DEBUG>=2
   int calls; //!< Number of hard decisions taken
   int ties; //!< Number of tie-breaks resolved
#endif
   // @}
public:
#if DEBUG>=2
   //! Default constructor
   basic_hard_decision() :
         calls(0), ties(0)
      {
      }
   //! Destructor
   ~basic_hard_decision()
      {
      if (calls > 0)
         std::cerr << "DEBUG (hard_decision): " << ties << " tie-breaks in "
               << calls << " hard decisions." << std::endl;
      }
#endif
   //! Seeds random generator from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }
   /*!
    * \brief Hard decision on soft information
    * \param[in] ri Likelihood table for input symbols
    * \return Index of the most likely input symbol
    *
    * Decide which input symbol was most probable. In case of ties, pick
    * randomly from tied values.
    */
   S operator()(const array1d_t& ri)
      {
#if DEBUG>=2
      calls++;
#endif
      // Inherit size
      const int K = ri.size();
      assert(K > 0);
      // Keep track of maximum value and list of indices
      dbl maxval = 0;
      std::list<int> indices;
      // Find list of indices with maximum value
      for (int i = 0; i < K; i++)
         if (ri(i) > maxval)
            {
            maxval = ri(i);
            indices.clear();
            indices.push_back(i);
            }
         else if (ri(i) == maxval)
            indices.push_back(i);
      // Return index of maximum value, if there is only one
      assert(indices.size() > 0);
      if (indices.size() == 1)
         return S(indices.front());
      // pick randomly in case of ties
#if DEBUG>=2
      ties++;
#endif
      std::list<int>::const_iterator it = indices.begin();
      const int skip = r.ival(indices.size());
      for (int i = 0; i < skip; i++)
         it++;
      return S(*it);
      }
};

template <template <class > class C, class dbl, class S>
class hard_decision : public basic_hard_decision<dbl, S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   void operator()(const C<array1d_t>& ri, C<S>& decoded);
};

template <class dbl, class S>
class hard_decision<libbase::vector, dbl, S> : public basic_hard_decision<dbl, S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   /*!
    * \brief Hard decision on soft information
    * \param[in]  ri       Likelihood table for input symbols at every timestep
    * \param[out] decoded  Sequence of the most likely input symbols at every
    * timestep
    *
    * Decide which input sequence was most probable.
    */
   void operator()(const libbase::vector<array1d_t>& ri,
         libbase::vector<S>& decoded)
      {
      // Determine sizes from input matrix
      const int tau = ri.size();
      assert(tau > 0);
#ifndef NDEBUG
      const int K = ri(0).size();
#endif
      // Initialise result vector
      decoded.init(tau);
      // Determine most likely symbol at every timestep
      for (int t = 0; t < tau; t++)
         {
         assert(ri(t).size() == K);
         decoded(t) = basic_hard_decision<dbl, S>::operator()(ri(t));
         }
      }
};

template <class dbl, class S>
class hard_decision<libbase::matrix, dbl, S> : public basic_hard_decision<dbl, S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   /*!
    * \brief Hard decision on soft information
    * \param[in]  ri       Likelihood table for input symbols at every timestep
    * \param[out] decoded  Sequence of the most likely input symbols at every
    * timestep
    *
    * Decide which input sequence was most probable.
    */
   void operator()(const libbase::matrix<array1d_t>& ri,
         libbase::matrix<S>& decoded)
      {
      // Determine sizes from input matrix
      const int rows = ri.size().rows();
      const int cols = ri.size().cols();
      assert(rows > 0 && cols > 0);
#ifndef NDEBUG
      const int K = ri(0, 0).size();
#endif
      // Initialise result vector
      decoded.init(rows, cols);
      // Determine most likely symbol at every timestep
      for (int i = 0; i < rows; i++)
         for (int j = 0; j < cols; j++)
            {
            assert(ri(i, j).size() == K);
            decoded(i, j) = basic_hard_decision<dbl, S>::operator()(ri(i, j));
            }
      }
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
