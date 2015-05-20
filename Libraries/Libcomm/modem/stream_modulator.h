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

#ifndef __stream_modulator_h
#define __stream_modulator_h

#include "informed_modulator.h"

namespace libcomm {

/*!
 * \brief   Stream-Oriented Modulator Interface.
 * \author  Johann Briffa
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
 * This interface is a superset of the informed blockmodem, defining:
 * - Vector demodulation methods that make use of prior frame-edge information
 *   and support look-ahead
 * - A method to get the posterior channel drift pdf at codeword boundaries
 * - A method to get the positions of codeword boundaries
 * - A getter for the suggested look-ahead
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
         const libbase::size_type<C> lookahead, const C<double>& sof_prior,
         const C<double>& eof_prior, const C<array1d_t>& app,
         C<array1d_t>& ptable, C<double>& sof_post, C<double>& eof_post,
         const libbase::size_type<C> offset) = 0;
   // @}

public:
   /*! \name Block modem operations - streaming extensions */
   /*!
    * \brief Demodulate a sequence of time-steps
    * \param[in]  chan     The channel model (used to obtain likelihoods)
    * \param[in]  rx       Sequence of received symbols
    * \param[in]  lookahead Number of modulation symbols beyond EOF supplied
    * \param[in]  sof_prior Prior probabilities for start-of-frame position
    *                      (zero-index matches zero-index of rx)
    * \param[in]  eof_prior Prior probabilities for end-of-frame position
    *                      (zero-index matches N-index of rx, where N is the
    *                      length of the transmitted frame + lookahead)
    * \param[in]  app      Prior probabilities of uncoded transmitted sequence
    * \param[out] ptable   Posterior probabilities of uncoded transmitted sequence
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
         const libbase::size_type<C> lookahead, const C<double>& sof_prior,
         const C<double>& eof_prior, const C<array1d_t>& app,
         C<array1d_t>& ptable, C<double>& sof_post, C<double>& eof_post,
         const libbase::size_type<C> offset)
      {
      this->advance_if_dirty();
      dodemodulate(chan, rx, lookahead, sof_prior, eof_prior, app, ptable,
            sof_post, eof_post, offset);
      this->mark_as_dirty();
      }
   /*!
    * \brief Get the posterior channel drift pdf at codeword boundaries
    * \param[out] pdftable Posterior Probabilities for codeword boundaries
    * \param[out] offset   Index offset for drift
    *
    * Codeword boundaries are taken to include frame boundaries, such that
    * pdftable(i) corresponds to the boundary between codewords 'i-1' and 'i',
    * where codewords are zero-indexed.
    *
    * This method must be called after a call to demodulate(), so that it can
    * return posteriors (and the offset used) for the last transmitted frame.
    */
   virtual void get_post_drift_pdf(C<array1d_t>& pdftable,
         libbase::size_type<C>& offset) const = 0;
   /*!
    * \brief Get the positions of codeword boundaries
    * \return Positions of codeword boundaries
    *
    * Codeword boundaries are taken to include frame boundaries.
    *
    * This method must be called after a call to modulate(), so that it can
    * return boundaries for the last transmitted frame.
    */
   virtual C<int> get_boundaries(void) const = 0;
   /*!
    * \brief Get the suggested look-ahead quantity
    * \return Number of modulation symbols to look into the next frame
    */
   virtual libbase::size_type<C> get_suggested_lookahead(void) const = 0;
   /*!
    * \brief Get the suggested state space exclusion probability
    * \return Probability of channel event outside chosen limits
    */
   virtual double get_suggested_exclusion(void) const = 0;
   // @}

   // Block modem operations
   // (necessary because inheriting methods from templated base)
   using Interface::modulate;
   using Interface::demodulate;
};

} // end namespace

#endif
