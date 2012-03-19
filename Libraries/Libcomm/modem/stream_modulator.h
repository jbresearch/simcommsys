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

#ifndef __stream_modulator_h
#define __stream_modulator_h

#include "informed_modulator.h"

namespace libcomm {

/*!
 * \brief   Stream-Oriented Modulator Interface.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Defines common interface for stream-oriented blockmodem classes. A stream-
 * oriented blockmodem is one which can determine frame synchronization and
 * can make use of a-priori information for start and end of frame positions.
 * It is also an 'informed' modulator in that it can use a-priori symbol
 * probabilities during the demodulation stage. In general, such a blockmodem
 * is used with synchronization-correcting codes and can also be used in an
 * iterative loop with the channel codec. Such a blockmodem can also determine
 * posterior probabilities for the start and end of frame positions.
 *
 * This interface is a superset of the informed blockmodem, defining new
 * (vector) demodulation methods that make use of prior frame-edge information.
 * Overloaded versions are provided to work with or without prior symbol
 * information, as well as to return (or not) posterior frame-edge information.
 */

template <class S, template <class > class C = libbase::vector>
class stream_modulator : public informed_modulator<S, C> {
private:
   // Shorthand for class hierarchy
   typedef informed_modulator<S, C> Interface;
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}
protected:
   /*! \name Interface with derived classes */
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S, C>& chan, const C<S>& rx,
         const C<double>& sof_prior, const C<double>& eof_prior, const C<
               array1d_t>& app, C<array1d_t>& ptable, C<double>& sof_post, C<
               double>& eof_post, const libbase::size_type<C> offset) = 0;
   // @}

public:
   /*! \name Block modem operations */
   /*!
    * \brief Demodulate a sequence of time-steps
    * \param[in]  chan     The channel model (used to obtain likelihoods)
    * \param[in]  rx       Sequence of received symbols
    * \param[in]  sof_prior Prior probabilities for start-of-frame position
    *                      (zero-index matches zero-index of rx)
    * \param[in]  eof_prior Prior probabilities for end-of-frame position
    *                      (zero-index matches N-index of rx, where N is the
    *                      length of the transmitted frame)
    * \param[in]  app      Prior probabilities of transmitted sequence
    * \param[out] ptable   Posterior probabilities of transmitted sequence
    * \param[out] sof_post Posterior probabilities for start-of-frame position
    *                      (zero-index matches zero-index of rx)
    * \param[out] eof_post Posterior probabilities for end-of-frame position
    *                      (zero-index matches N-index of rx, where N is the
    *                      length of the transmitted frame)
    * \param[in]  offset   Index offset for prior, post, and rx vectors
    *
    * \note To avoid the need for a container class that can take a non-zero
    *       index, the user needs to supply an offset value. This is needed for
    *       both prior and post containers as well as for the rx container.
    *       The offsets are defined in a positive way, so that the physical
    *       index is equal to the virtual index + offset.
    *
    * \todo Remove offset and replace vector types with ones that can take a
    *       non-zero index.
    *
    * \note \c ptable(i,d) \c is the posterior probability of having transmitted
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
         const C<double>& sof_prior, const C<double>& eof_prior, const C<
               array1d_t>& app, C<array1d_t>& ptable, C<double>& sof_post, C<
               double>& eof_post, const libbase::size_type<C> offset)
      {
      this->advance_if_dirty();
      dodemodulate(chan, rx, sof_prior, eof_prior, app, ptable, sof_post,
            eof_post, offset);
      this->mark_as_dirty();
      }
   // @}

   // Block modem operations
   // (necessary because inheriting methods from templated base)
   using Interface::modulate;
   using Interface::demodulate;
};

} // end namespace

#endif
