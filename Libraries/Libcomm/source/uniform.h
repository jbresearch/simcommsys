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

#ifndef __source_uniform_h
#define __source_uniform_h

#include "config.h"
#include "source.h"
#include "serializer.h"
#include "field_utils.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief   Uniform random source.
 * \author  Johann Briffa
 *
 * Implements a source that returns a uniformly-distributed random symbol
 * from the given alphabet.
 */

template <class S, template <class > class C = libbase::vector>
class uniform : public source<S, C> {
private:
   /*! \name Internal representation */
   libbase::randgen r; //!< Data sequence generator
   // @}
public:
   //! Generate a single source element
   S generate_single()
      {
      return r.ival(field_utils<S>::elements());
      }

   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   //! Description
   std::string description() const
      {
      std::ostringstream sout;
      sout << field_utils<S>::elements() << "-ary uniform random source";
      return sout.str();
      }

   // Serialization Support
DECLARE_SERIALIZER(uniform)
};

/*!
 * \brief   Uniform random source specialisation.
 * \author  Johann Briffa
 *
 * Implements a source that returns a uniformly-distributed random symbol
 * from the given alphabet. Partial specialisation for int (needs to obtain
 * alphabet size from serialization).
 */

template <template <class > class C>
class uniform<int, C> : public source<int, C> {
private:
   /*! \name Internal representation */
   libbase::randgen r; //!< Data sequence generator
   int alphabet_size; //!< Number of elements in alphabet
   // @}
public:
   //! Constructor
   uniform(int alphabet_size=0) : alphabet_size(alphabet_size)
      {
      assert(alphabet_size >= 0);
      }

   //! Generate a single source element
   int generate_single()
      {
      return r.ival(alphabet_size);
      }

   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   //! Description
   std::string description() const
      {
      std::ostringstream sout;
      sout << alphabet_size << "-ary uniform random source";
      return sout.str();
      }

   // Serialization Support
DECLARE_SERIALIZER(uniform)
};

} // end namespace

#endif

