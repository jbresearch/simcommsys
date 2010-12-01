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

#ifndef __qsc_h
#define __qsc_h

#include "config.h"
#include "channel.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   q-ary symmetric channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \version 1.00 (12 Feb 2008)
 * - Initial version; implementation of a q-ary symmetric channel as
 * templated class.
 *
 * \version 1.01 (13 Feb 2008)
 * - Fixed check on range of Ps
 * - Changed assert to assertalways in range check
 * - Fixed PDF result for erroneous symbols
 *
 * \version 1.02 (16 Apr 2008)
 * - Fixed computation in corrupt() to force addition within the field
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
   double pdf(const G& tx, const G& rx) const;
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   qsc()
      {
      }
   // @}

   /*! \name Channel parameter handling */
   //! Set the substitution probability
   void set_parameter(const double Ps);
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

template <class G>
inline double qsc<G>::pdf(const G& tx, const G& rx) const
   {
   return (tx == rx) ? 1 - Ps : Ps / G::elements();
   }

} // end namespace

#endif

