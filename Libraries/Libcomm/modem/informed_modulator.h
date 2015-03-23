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

#ifndef __informed_modulator_h
#define __informed_modulator_h

#include "blockmodem.h"

namespace libcomm {

/*!
 * \brief   Informed Modulator Interface.
 * \author  Johann Briffa
 *
 * Defines common interface for informed blockmodem classes. An informed
 * blockmodem is one which can use a-priori symbol probabilities during the
 * demodulation stage. In general, such a blockmodem may be used in an iterative
 * loop with the channel codec.
 *
 * This interface is a superset of the regular blockmodem, defining two new
 * demodulation methods (atomic and vector) that make use of prior information.
 *
 * \todo Figure out whether atomic modem operations are really used anywhere
 */

template <class S, template <class > class C = libbase::vector>
class informed_modulator : public blockmodem<S, C> {
public:
   /*! \name Type definitions */
   typedef blockmodem<S, C> Base;
   typedef libbase::vector<double> array1d_t;
   // @}
protected:
   /*! \name Interface with derived classes */
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S, C>& chan, const C<S>& rx,
         const C<array1d_t>& app, C<array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Atomic modem operations - informed extensions */
   /*!
    * \brief Demodulate a single time-step
    * \param[in]  signal   Received signal
    * \param[in]  app      Table of a-priori likelihoods of possible
    * transmitted symbols
    * \return  Index corresponding symbol that is closest to the received signal
    */
   virtual const int demodulate(const S& signal, const array1d_t& app) const = 0;
   // @}

   /*! \name Block modem operations - informed extensions */
   /*!
    * \brief Demodulate a sequence of time-steps
    * \param[in]  chan     The channel model (used to obtain likelihoods)
    * \param[in]  rx       Sequence of received symbols
    * \param[in]  app      Table of a-priori likelihoods of possible
    * transmitted symbols at every time-step
    * \param[out] ptable   Table of likelihoods of possible transmitted symbols
    *
    * \note \c ptable(i,d) \c is the a posteriori probability of having transmitted
    * symbol 'd' at time 'i'
    *
    * \note This function is non-const, to support time-variant modulation
    * schemes such as DM inner codes.
    *
    * \note app and ptable may point to the same space
    *
    * \note app may be empty; this should be taken to indicate that no prior
    * information is available
    */
   void demodulate(const channel<S, C>& chan, const C<S>& rx,
         const C<array1d_t>& app, C<array1d_t>& ptable)
      {
      this->advance_if_dirty();
      dodemodulate(chan, rx, app, ptable);
      this->mark_as_dirty();
      }
   // @}

   // Block modem operations
   // (necessary because overloaded methods hide those in templated base)
   using Base::modulate;
   using Base::demodulate;
};

} // end namespace

#endif
