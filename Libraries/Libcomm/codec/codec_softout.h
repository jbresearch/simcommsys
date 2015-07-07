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

#ifndef __codec_softout_h
#define __codec_softout_h

#include "config.h"
#include "codec.h"
#include "hard_decision.h"

namespace libcomm {

/*!
 * \brief   Channel Codec with Soft Output.
 * \author  Johann Briffa
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_softout: public codec<C, dbl> {
private:
   // Shorthand for class hierarchy
   typedef codec<C, dbl> Base;

public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

private:
   /*! \name Internally-used objects */
   hard_decision<C, dbl, int> hd_functor; //!< Hard-decision box
   // @}

protected:
   /*! \name Interface with derived classes */
   //! \copydoc init_decoder()
   virtual void do_init_decoder(const C<array1d_t>& ptable,
         const C<array1d_t>& app) = 0;
   // @}
public:
   /*! \name Codec operations */
   void seedfrom(libbase::random& r)
      {
      // Call base method first
      Base::seedfrom(r);
      // Seed hard-decision box
      hd_functor.seedfrom(r);
      }
   void decode(C<int>& decoded)
      {
      libbase::cputimer t("t_decode");
      C<array1d_t> ri;
      this->softdecode(ri);
      hd_functor(ri, decoded);
      this->add_timer(t);
      }
   // Inherit receiver translation process from base class
   using Base::init_decoder;
   /*!
    * \brief Receiver translation process (with given priors)
    * \param[in] ptable Likelihoods of each possible encoded symbol at every index
    * \param[in] app Likelihoods of each possible input symbol at every index
    *
    * This function initializes the decoder with the probability tables for
    * each encoded symbol as received from the blockmodem.
    * This function should be called before the first decode iteration
    * for each block.
    */
   void init_decoder(const C<array1d_t>& ptable, const C<array1d_t>& app)
      {
      libbase::cputimer t("t_init_decoder");
      this->advance_if_dirty();
      do_init_decoder(ptable, app);
      this->mark_as_dirty();
      this->add_timer(t);
      }
   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    *
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   virtual void softdecode(C<array1d_t>& ri) = 0;
   /*!
    * \brief Decoding process
    * \param[out] ri Likelihood table for input symbols at every timestep
    * \param[out] ro Likelihood table for output symbols at every timestep
    *
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   virtual void softdecode(C<array1d_t>& ri, C<array1d_t>& ro) = 0;
   // @}
};

} // end namespace

#endif
