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

#ifndef __map_stipple_h
#define __map_stipple_h

#include "map_straight.h"

namespace libcomm {

/*!
 * \brief   Stipple Mapper.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class defines an punctured version of the straight mapper, suitable
 * for turbo codes, where:
 * - all information symbols are transmitted
 * - parity symbols are taken from successive sets to result in an overall
 * rate of 1/2
 * For a two-set turbo code, this corresponds to odd/even puncturing.
 *
 * \note This supersedes the puncture_stipple class; observe though that the
 * number of sets here corresponds to the definition used in the turbo
 * code, and is one less than that for puncture_stipple.
 *
 * \bug This is really only properly defined for vector containers.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_stipple : public map_straight<C, dbl> {
private:
   // Shorthand for class hierarchy
   typedef map_straight<C, dbl> Base;
   typedef map_stipple<C, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

private:
   /*! \name User-defined parameters */
   int sets; //!< Number of turbo code parallel sets
   // @}
   /*! \name Internal object representation */
   mutable C<bool> pattern; //!< Pre-computed puncturing pattern
   // @}

protected:
   // Pull in base class variables
   using Base::size;
   using Base::M;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

public:
   /*! \name Constructors / Destructors */
   virtual ~map_stipple()
      {
      }
   // @}

   // Informative functions
   double rate() const
      {
      return (sets + 1) / 2.0;
      }
   libbase::size_type<C> output_block_size() const
      {
      const int tau = size.length() / (sets + 1);
      assert(size.length() == tau * (sets + 1));
      return libbase::size_type<C>(tau * 2);
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_stipple)
};

} // end namespace

#endif
