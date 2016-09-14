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

#ifndef __direct_blockmodem_h
#define __direct_blockmodem_h

#include "blockmodem.h"
#include "direct_modem.h"

namespace libcomm {

/*!
 * \brief   Q-ary Blockwise Modulator.
 * \author  Johann Briffa
 *
 * This class is a template definition for q-ary channel block modems; this
 * needs to be specialized for actual use. Template parameter defaults are
 * provided here.
 */

template <class G, template <class > class C = libbase::vector,
      class dbl = double>
class direct_blockmodem : public blockmodem<G, C, dbl>,
      protected direct_modem<G> {
};

/*!
 * \brief   Q-ary Blockwise Modulator Vector Implementation.
 * \author  Johann Briffa
 *
 * Vector implementation of plain q-ary channel modulation.
 *
 * \todo Merge modulate and demodulate between this function and lut_modulator
 */

template <class G, class dbl>
class direct_blockmodem<G, libbase::vector, dbl> : public blockmodem<G,
      libbase::vector, dbl>, protected direct_modem<G> {
public:
   /*! \name Type definitions */
   typedef direct_modem<G> Implementation;
   typedef libbase::vector<dbl> array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded,
         libbase::vector<G>& tx);
   void dodemodulate(const channel<G, libbase::vector>& chan,
         const libbase::vector<G>& rx, libbase::vector<array1d_t>& ptable);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(direct_blockmodem)
};

} // end namespace

#endif
