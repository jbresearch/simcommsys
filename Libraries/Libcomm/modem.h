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

#ifndef __modem_h
#define __modem_h

#include "config.h"
#include "random.h"
#include "sigspace.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Common Modulator Interface.
 * \author  Johann Briffa
 *
 * Class defines common interface for modem classes.
 */

template <class S>
class basic_modem {
public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_modem()
      {
      }
   // @}

   /*! \name Atomic modem operations */
   /*!
    * \brief Modulate a single time-step
    * \param   index Index into the symbol alphabet
    * \return  Symbol corresponding to the given index
    */
   virtual const S modulate(const int index) const = 0;
   /*!
    * \brief Demodulate a single time-step
    * \param   signal   Received signal
    * \return  Index corresponding symbol that is closest to the received signal
    */
   virtual const int demodulate(const S& signal) const = 0;
   /*! \copydoc modulate() */
   const S operator[](const int index) const
      {
      return modulate(index);
      }
   /*! \copydoc demodulate() */
   const int operator[](const S& signal) const
      {
      return demodulate(signal);
      }
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   // @}

   /*! \name Informative functions */
   //! Symbol alphabet size at input
   virtual int num_symbols() const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 * \brief   Modulator Base.
 * \author  Johann Briffa
 */

template <class S>
class modem : public basic_modem<S> {
};

/*!
 * \brief   Signal-Space Modulator Specialization.
 * \author  Johann Briffa
 */

template <>
class modem<sigspace> : public basic_modem<sigspace> {
public:
   /*! \name Informative functions */
   //! Average energy per symbol
   virtual double energy() const = 0;
   //! Average energy per bit
   double bit_energy() const
      {
      return energy() / log2(num_symbols());
      }
   //! Modulation rate (spectral efficiency) in bits/unit energy
   double rate() const
      {
      return 1.0 / bit_energy();
      }
   // @}
};

} // end namespace

#endif
