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

#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
 * \brief   Straight Mapper Template.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class is a template definition for straight mappers; this needs to
 * be specialized for actual use. Template parameter defaults are provided
 * here.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_straight : public mapper<C, dbl> {
};

/*!
 * \brief   Straight Mapper - Vector containers.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class defines a straight symbol mapper with:
 * forward transform from blockmodem
 * inverse transform from the various codecs.
 */

template <class dbl>
class map_straight<libbase::vector, dbl> : public mapper<libbase::vector, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector, dbl> Base;
   typedef map_straight<libbase::vector, dbl> This;

private:
   /*! \name Internal object representation */
   int s1; //!< Number of modulation symbols per encoder output
   int s2; //!< Number of modulation symbols per translation symbol
   int upsilon; //!< Block size in symbols at codec translation
   // @}

protected:
   // Pull in base class variables
   using Base::size;
   using Base::M;
   using Base::N;
   using Base::S;

protected:
   // Interface with mapper
   void setup();
   void dotransform(const array1i_t& in, array1i_t& out) const;
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const;

public:
   // Informative functions
   double rate() const
      {
      return 1;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(size * s1);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_straight)
};

/*!
 * \brief   Straight Mapper - Matrix containers.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class defines a straight symbol mapper, where it is assumed that:
 * - the input and output alphabet sizes are the same
 * - matrix reshaping occurs by reading and writing elements in row-major
 * order
 */

template <class dbl>
class map_straight<libbase::matrix, dbl> : public mapper<libbase::matrix, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::matrix<array1d_t> array2vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::matrix, dbl> Base;
   typedef map_straight<libbase::matrix, dbl> This;

private:
   /*! \name Internal object representation */
   libbase::size_type<libbase::matrix> size_out;
   // @}

protected:
   // Pull in base class variables
   using Base::size;
   using Base::M;
   using Base::N;
   using Base::S;

protected:
   // Interface with mapper
   void setup();
   void dotransform(const array2i_t& in, array2i_t& out) const;
   void doinverse(const array2vd_t& pin, array2vd_t& pout) const;

public:
   // Informative functions
   double rate() const
      {
      return 1;
      }
   libbase::size_type<libbase::matrix> output_block_size() const
      {
      return size_out;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_straight)
};

} // end namespace

#endif
