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

#ifndef __offset_vector_h
#define __offset_vector_h

#include "config.h"
#include "vector.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of memory allocation/deallocation
// 3 - Trace memory allocation/deallocation
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Offset Vector
 * \author  Johann Briffa
 *
 * An offset vector is a vector with a user-selected base index.
 */

template <class T>
class offset_vector : public vector<T> {
   typedef vector<T> Base;
protected:
   size_type<libbase::vector> m_offset;
public:
   /*! \name Other Constructors */
   /*! \brief Default constructor
    * Allocates space for 'n' elements, with the index to the first element
    * equal to 'base'. Note that this does not initialize elements.
    */
   explicit offset_vector(const int n = 0, const int base = 0) :
      Base(n), m_offset(base)
      {
      }
   /*! \brief On-the-fly conversion of vectors
    * Copies the elements of the given vector and resets the index to the first
    * element to 'base'.
    * \note Naturally this requires a deep copy.
    */
   template <class A>
   explicit offset_vector(const vector<A>& x, const int base = 0) :
      Base(x), m_offset(base)
      {
      }
   // @}

   /*! \name Resizing operations */
   /*! \brief Set index to the first element to 'base'
    */
   void set_base(const int base)
      {
      m_offset = base;
      }
   // @}

   /*! \name Element access */
   /*! \brief Index operator (write-access)
    * \note Performs boundary checking.
    */
   T& operator()(const int x)
      {
      test_invariant();
      return Base(x - base);
      }
   /*! \brief Index operator (read-only access)
    * \note Performs boundary checking.
    */
   const T& operator()(const int x) const
      {
      test_invariant();
      return Base(x - base);
      }
   // @}

   // information services
   //! Base index
   size_type<libbase::vector> get_base() const
      {
      return m_offset;
      }
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
