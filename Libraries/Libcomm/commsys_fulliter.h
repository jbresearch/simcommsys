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

#ifndef __commsys_fulliter_h
#define __commsys_fulliter_h

#include "commsys.h"

namespace libcomm {

/*!
 * \brief   Communication System with Full-System Iteration.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Communication system with iterative demodulation and decoding; the model here
 * is such that we demodulate once, then decode for M iterations. After that,
 * we pass the posterior information as prior information for a second
 * demodulation, followed again by M decoding iterations. This is repeated for
 * N demodulations (ie. full-system iterations), giving a total of N.M results.
 *
 * \note This only works with straight mapping for now.
 *
 * \todo Integrate this nature within updated commsys interface.
 *
 * \todo Update mapper interface as necessary to allow this to work with
 * non-straight mappers
 */

template <class S, template <class > class C = libbase::vector>
class commsys_fulliter : public commsys<S, C> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef commsys<S, C> Base;
   typedef commsys_fulliter<S, C> This;
private:
   /*! \name User parameters */
   int iter; //!< Number of full-system iterations
   // @}
   /*! \name Internal state */
   int cur_cdc_iter; //!< Current decoder iteration
   int cur_mdm_iter; //!< Current modem iteration
   C<S> last_received; //!< Last received block
   C<array1d_t> ptable_mapped; //!< Prior information to use in demodulation
   // @}
protected:
   /*! \name Helper functions */
   void compute_extrinsic(C<array1d_t>& re, const C<array1d_t>& ro, const C<
         array1d_t>& ri);
   // @}
public:
   // Communication System Interface
   void receive_path(const C<S>& received);
   void decode(C<int>& decoded);
   // Informative functions
   int num_iter() const
      {
      return this->cdc->num_iter() * iter;
      }

   // Description
   std::string description() const;
   // Serialization Support
DECLARE_SERIALIZER(commsys_fulliter)
};

} // end namespace

#endif
