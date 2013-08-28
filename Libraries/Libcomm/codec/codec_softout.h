/*!
 * \file
 * $Id$
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
 * \brief   Channel Codec with Soft Output Interface.
 * \author  Johann Briffa
 * $Id$
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_softout_interface : public codec<C, dbl> {
private:
   // Shorthand for class hierarchy
   typedef codec<C, dbl> Base;

public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

protected:
   /*! \name Internal codec operations */
   /*!
    * \brief A-priori probability initialization
    *
    * This function resets the a-priori prabability tables for the codec to
    * equally-likely. This function (or setpriors) should be called before the
    * first decode iteration for each block.
    */
   virtual void resetpriors() = 0;
   /*!
    * \brief A-priori probability setup
    * \param[in] ptable Likelihoods of each possible input symbol at every
    * (input) timestep
    *
    * This function updates the a-priori prabability tables for the codec.
    * This function (or resetpriors) should be called before the first decode
    * iteration for each block.
    */
   virtual void setpriors(const C<array1d_t>& ptable) = 0;
   /*!
    * \copydoc codec::init_decoder()
    *
    * \note Sets up receiver likelihood tables only.
    */
   virtual void setreceiver(const C<array1d_t>& ptable) = 0;
   // @}
protected:
   /*! \name Interface with derived classes */
   //! \copydoc init_decoder()
   virtual void do_init_decoder(const C<array1d_t>& ptable,
         const C<array1d_t>& app) = 0;
   // @}
public:
   /*! \name Codec operations */
   using Base::init_decoder;
   /*!
    * \copydoc codec::init_decoder()
    * \param[in] app Likelihoods of each possible input symbol at every
    * (input) timestep
    */
   void init_decoder(const C<array1d_t>& ptable, const C<array1d_t>& app)
      {
      //libbase::cputimer t("t_init_decoder");
      this->advance_if_dirty();
      do_init_decoder(ptable, app);
      this->mark_as_dirty();
      //add_timer(t);
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

/*!
 * \brief   Channel Codec with Soft Output Base.
 * \author  Johann Briffa
 * $Id$
 *
 * Templated soft-output codec base. This extra level is required to allow
 * partial specialization of the container.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_softout : public codec_softout_interface<C, dbl> {
public:
};

/*!
 * \brief   Channel Codec with Soft Output Base Specialization.
 * \author  Johann Briffa
 * $Id$
 *
 * Templated soft-output codec base. This extra level is required to allow
 * partial specialization of the container.
 */

template <class dbl>
class codec_softout<libbase::vector, dbl> : public codec_softout_interface<
      libbase::vector, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   /*! \name Internally-used objects */
   hard_decision<libbase::vector, dbl, int> hd_functor; //!< Hard-decision box
   // @}
protected:
   // Interface with derived classes
   void do_init_decoder(const array1vd_t& ptable)
      {
      array1vd_t temp;
      temp = ptable;
      this->setreceiver(temp);
      this->resetpriors();
      }
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app)
      {
      this->setreceiver(ptable);
      this->setpriors(app);
      }
public:
   // Codec operations
   void seedfrom(libbase::random& r)
      {
      // Call base method first
      codec_softout_interface<libbase::vector, dbl>::seedfrom(r);
      // Seed hard-decision box
      hd_functor.seedfrom(r);
      }
   void decode(array1i_t& decoded)
      {
      array1vd_t ri;
      this->softdecode(ri);
      hd_functor(ri, decoded);
      }
};

} // end namespace

#endif
