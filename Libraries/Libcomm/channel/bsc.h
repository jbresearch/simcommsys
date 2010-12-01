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

#ifndef __bsc_h
#define __bsc_h

#include "config.h"
#include "channel.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   Binary symmetric channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \version 1.00 (25 Jan 2008)
 * - Initial version; implementation of a binary symmetric channel.
 */

class bsc : public channel<bool> {
private:
   /*! \name User-defined parameters */
   double Ps; //!< Bit-substitution probability \f$ P_s \f$
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s);
   double pdf(const bool& tx, const bool& rx) const;
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   bsc()
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
DECLARE_SERIALIZER(bsc)
};

inline double bsc::pdf(const bool& tx, const bool& rx) const
   {
   return (tx != rx) ? Ps : 1 - Ps;
   }

} // end namespace

#endif

