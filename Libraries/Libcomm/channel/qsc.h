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

#ifndef __qsc_h
#define __qsc_h

#include "config.h"
#include "channel.h"
#include "field_utils.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   q-ary symmetric channel.
 * \author  Johann Briffa
 *
 * Implements a q-ary symmetric channel as a templated class.
 */

template <class G>
class qsc : public channel<G> {
private:
   /*! \name User-defined parameters */
   double Ps; //!< Symbol-substitution probability \f$ P_s \f$
   // @}
protected:
   // Channel function overrides
   G corrupt(const G& s);
   double pdf(const G& tx, const G& rx) const
      {
      return (tx == rx) ? 1 - Ps : Ps / field_utils<G>::elements();
      }
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   qsc()
      {
      }
   // @}

   /*! \name Channel parameter handling */
   //! Set the substitution probability
   void set_parameter(const double Ps)
      {
      const double q = field_utils<G>::elements();
      assertalways(Ps >=0 && Ps <= (q-1)/q);
      qsc::Ps = Ps;
      }
   //! Get the substitution probability
   double get_parameter() const
      {
      return Ps;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qsc)
};

} // end namespace

#endif

