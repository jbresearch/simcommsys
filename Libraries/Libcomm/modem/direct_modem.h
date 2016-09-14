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

#ifndef __direct_modem_h
#define __direct_modem_h

#include "modem.h"

namespace libcomm {

/*!
 * \brief   Q-ary Modulator.
 * \author  Johann Briffa
 *
 * Specific implementation of q-ary channel modulation.
 *
 * \note Template argument class must provide a method elements() that returns
 * the field size.
 *
 * \todo Merge modulate and demodulate between this function and lut_modulator (?)
 *
 * \todo Require symbol class to provide description() method
 */

template <class G>
class direct_modem : public virtual modem<G> {
public:
   // Atomic modem operations
   const G modulate(const int index) const
      {
      assert(index >= 0 && index < num_symbols());
      return G(index);
      }
   const int demodulate(const G& signal) const
      {
      return signal;
      }

   // Informative functions
   int num_symbols() const
      {
      return G::elements();
      }

   // Description
   std::string description() const;
};

/*!
 * \brief   Binary Modulator Specialization.
 * \author  Johann Briffa
 *
 * Specific implementation of binary channel modulation.
 */

template <>
class direct_modem<bool> : public virtual modem<bool> {
public:
   // Atomic modem operations
   const bool modulate(const int index) const
      {
      assert(index >= 0 && index <= 1);
      return index & 1;
      }
   const int demodulate(const bool& signal) const
      {
      return signal;
      }

   // Informative functions
   int num_symbols() const
      {
      return 2;
      }

   // Description
   std::string description() const;
};

} // end namespace

#endif
