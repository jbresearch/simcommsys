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

#ifndef __qec_h
#define __qec_h

#include "config.h"
#include "channel.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   q-ary erasure channel.
 * \author  Johann Briffa
 *
 * Implements an erasure channel; the symbol type is specified as a template
 * parameter. The symbol type must support the following methods:
 * - An elements() methods that returns the alphabet size
 * - An erase() method that marks the symbol as erased
 * - An is_erased() method that queries the erasure status of the symbol
 * - An operator==() comparison between two symbols of that type
 * Typically this will be provided by the erasable<> templated class.
 */

template <class G>
class qec : public channel<G> {
private:
   /*! \name User-defined parameters */
   double Pe; //!< Symbol-erasure probability \f$ P_e \f$
   // @}
protected:
   // Channel function overrides
   G corrupt(const G& s);
   double pdf(const G& tx, const G& rx) const
      {
      // handle erasures first
      if (rx.is_erased())
         return 1.0 / G::elements();
      // otherwise the only option with non-zero probability is if we
      // received the same symbol that was transmitted
      if (tx == rx)
         return 1;
      return 0;
      }
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   qec()
      {
      }
   // @}

   /*! \name Channel parameter handling */
   //! Set the erasure probability
   void set_parameter(const double Pe)
      {
      assertalways(Pe >=0 && Pe <= 1);
      qec::Pe = Pe;
      }
   //! Get the erasure probability
   double get_parameter() const
      {
      return Pe;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qec)
};

} // end namespace

#endif

