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
 * \brief   Q-ary Blockwise Modulator Implementation.
 * \author  Johann Briffa
 *
 * This class is a template definition for q-ary channel block modems; this
 * needs to be specialized for actual use. Template parameter defaults are
 * provided here.
 */

template <class G, template <class > class C = libbase::vector,
      class dbl = double>
class direct_blockmodem_implementation : public direct_modem_implementation<G> {
};

/*!
 * \brief   Q-ary Blockwise Modulator Vector Implementation.
 * \author  Johann Briffa
 *
 * Vector implementation of plain q-ary channel modulation.
 */

template <class G, class dbl>
class direct_blockmodem_implementation<G, libbase::vector, dbl> : public direct_modem_implementation<
      G> {
public:
   /*! \name Type definitions */
   typedef direct_modem_implementation<G> Implementation;
   typedef libbase::vector<dbl> array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded,
         libbase::vector<G>& tx);
   void dodemodulate(const channel<G, libbase::vector>& chan,
         const libbase::vector<G>& rx, libbase::vector<array1d_t>& ptable);
};

/*!
 * \brief   Q-ary Blockwise Modulator.
 * \author  Johann Briffa
 *
 * Implementation of plain q-ary channel modulation.
 *
 * \todo Merge modulate and demodulate between this function and lut_modulator
 *
 * \todo Find out why using declarations are not working.
 */

template <class G, template <class > class C = libbase::vector,
      class dbl = double>
class direct_blockmodem : public blockmodem<G, C, dbl> ,
      protected direct_blockmodem_implementation<G, C, dbl> {
public:
   /*! \name Type definitions */
   typedef direct_blockmodem_implementation<G, C, dbl> Implementation;
   typedef libbase::vector<dbl> array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const C<int>& encoded, C<G>& tx)
      {
      // Check validity
      assertalways(encoded.size() == this->input_block_size());
      assertalways(N == this->num_symbols());
      // De-reference
      Implementation::domodulate(N, encoded, tx);
      }
   void dodemodulate(const channel<G, C>& chan, const C<G>& rx,
         C<array1d_t>& ptable)
      {
      // Check validity
      assertalways(rx.size() == this->input_block_size());
      // De-reference
      Implementation::dodemodulate(chan, rx, ptable);
      }

public:
   // Use implementation from base
   // Atomic modem operations
   const G modulate(const int index) const
      {
      return Implementation::modulate(index);
      }
   const int demodulate(const G& signal) const
      {
      return Implementation::demodulate(signal);
      }
   // Informative functions
   int num_symbols() const
      {
      return Implementation::num_symbols();
      }
   //using Implementation::modulate;
   //using Implementation::demodulate;
   //using Implementation::num_symbols;

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(direct_blockmodem)
};

} // end namespace

#endif
