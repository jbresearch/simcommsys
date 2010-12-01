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

#ifndef __lut_modulator_h
#define __lut_modulator_h

#include "blockmodem.h"

namespace libcomm {

/*!
 * \brief   LUT Modulator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class lut_modulator : public blockmodem<sigspace> {
public:
   /*! \name Type definitions */
   typedef blockmodem<sigspace> Base;
   typedef libbase::vector<double> array1d_t;
   // @}

protected:
   libbase::vector<sigspace> lut; // Array of modulation symbols

protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded,
         libbase::vector<sigspace>& tx);
   void dodemodulate(const channel<sigspace>& chan, const libbase::vector<
         sigspace>& rx, libbase::vector<array1d_t>& ptable);

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~lut_modulator()
      {
      }
   // @}

   // Atomic modem operations
   const sigspace modulate(const int index) const
      {
      return lut(index);
      }
   const int demodulate(const sigspace& signal) const;

   // Block modem operations
   // (necessary because overloaded methods hide those in templated base)
   using Base::modulate;
   using Base::demodulate;

   // Informative functions
   int num_symbols() const
      {
      return lut.size();
      }
   double energy() const; // average energy per symbol
};

} // end namespace

#endif
