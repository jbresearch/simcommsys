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

#ifndef __codec_softout_h
#define __codec_softout_h

#include "config.h"
#include "codec.h"

namespace libcomm {

/*!
 * \brief   Channel Codec with Soft Output Interface.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_softout_interface : public codec<C, dbl> {
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
public:
   /*! \name Codec operations */
   /*!
    * \copydoc codec::init_decoder()
    * \param[in] app Likelihoods of each possible input symbol at every
    * (input) timestep
    */
   virtual void init_decoder(const C<array1d_t>& ptable,
         const C<array1d_t>& app) = 0;
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
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
public:
   // Codec operations
   void init_decoder(const array1vd_t& ptable);
   void init_decoder(const array1vd_t& ptable, const array1vd_t& app);
   void decode(array1i_t& decoded);
};

} // end namespace

#endif
