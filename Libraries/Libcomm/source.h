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

#ifndef __source_h
#define __source_h

#include "config.h"
#include "serializer.h"
#include "vector.h"
#include "matrix.h"
#include "instrumented.h"

#include "randgen.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Common Source Interface.
 * \author  Johann Briffa
 *
 * Base source definition provides source model defined by generate methods,
 * one that returns a single element and one that fills a given container.
 */

template <class S, template <class > class C>
class basic_source_interface : public instrumented {
public:
   /*! \name Constructors / Destructors */
   virtual ~basic_source_interface()
      {
      }
   // @}

   /*! \name Source generation interface */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   //! Generate a single source element
   virtual S generate_single() = 0;
   /*!
    * \brief Generate a source sequence of the required size
    * \param[in]  blocksize     Required sequence size
    * \param[out] source        Generated source sequence
    */
   virtual C<S> generate_sequence(const libbase::size_type<C>& blocksize) = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 * \brief   Common Source Base.
 * \author  Johann Briffa
 *
 * Templated common source base. This extra level is required to allow partial
 * specialization of the container.
 */

template <class S, template <class > class C>
class basic_source : public basic_source_interface<S, C> {
};

/*!
 * \brief   Common Source Base Specialization.
 * \author  Johann Briffa
 *
 * Templated common source base. Partial specialization for vector container.
 * Provides default implementation to generate a sequence of elements using
 * the single-element generator.
 */

template <class S>
class basic_source<S, libbase::vector> : public basic_source_interface<S,
      libbase::vector> {
public:
   libbase::vector<S> generate_sequence(
         const libbase::size_type<libbase::vector>& blocksize)
      {
      // allocate space
      libbase::vector<S> source_sequence(blocksize);
      // fill as required
      for (int i = 0; i < source_sequence.size(); i++)
         source_sequence(i) = this->generate_single();
      return source_sequence;
      }
};


/*!
 * \brief   Common Channel Base Specialization.
 * \author  Johann Briffa
 *
 * Templated common source base. Partial specialization for matrix container.
 * Provides default implementation to generate a sequence of elements using
 * the single-element generator. Elements are stored in the matrix in row-major
 * order.
 */

template <class S>
class basic_source<S, libbase::matrix> : public basic_source_interface<S,
      libbase::matrix> {
public:
   libbase::vector<S> generate_sequence(
         const libbase::size_type<libbase::matrix>& blocksize)
      {
      // allocate space
      libbase::matrix<S> source_sequence(blocksize);
      // fill as required
      for (int i = 0; i < source_sequence.size().rows(); i++)
         for (int j = 0; j < source_sequence.size().cols(); j++)
            source_sequence(i, j) = this->generate_single();
      return source_sequence;
      }
};

/*!
 * \brief   Channel Base.
 * \author  Johann Briffa
 *
 * Templated base source model.
 */

template <class S, template <class > class C = libbase::vector>
class source : public basic_source<S, C> , public libbase::serializable {
   // Serialization Support
DECLARE_BASE_SERIALIZER(source)
};


} // end namespace

#endif
