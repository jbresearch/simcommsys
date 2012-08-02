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

#ifndef __dminner2_h
#define __dminner2_h

#include "config.h"

#include "dminner.h"
#ifdef USE_CUDA
#  include "algorithm/fba2-cuda.h"
#  include "dminner2-receiver-cuda.h"
#else
#  include "algorithm/fba2.h"
#  include "dminner2-receiver.h"
#endif

namespace libcomm {

/*!
 * \brief   Davey-MacKay Inner Code, with symbol-level decoding.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements a novel (and more accurate) decoding algorithm for the inner
 * codes described by Davey and MacKay in "Reliable Communication over Channels
 * with Insertions, Deletions, and Substitutions", Trans. IT, Feb 2001.
 */

template <class real>
class dminner2 : public dminner<real> {
private:
   // Shorthand for class hierarchy
   typedef stream_modulator<bool> Interface;
   typedef dminner2<real> This;
   typedef dminner<real> Base;
public:
   /*! \name Type definitions */
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   // @}
private:
   /*! \name User-defined parameters */
   bool batch; //!< Flag indicating use of batch receiver interface
   bool lazy; //!< Flag indicating lazy computation of gamma metric
   bool globalstore; //!< Flag indicating we will try to cache lazily computed gamma values
   // @}
   /*! \name Internally-used objects */
   // algorithm object
#ifdef USE_CUDA
   cuda::fba2<cuda::dminner2_receiver<real>, bool, real> fba;
#else
   fba2<dminner2_receiver<real> , bool, real> fba;
#endif
   // @}
private:
   //! Set up for given channel parameters and sof prior
   void init(const channel<bool>& chan, const array1d_t& sof_pdf,
         const int offset);
   //! Set up for given channel parameters and known start
   void init(const channel<bool>& chan)
      {
      const array1d_t eof_pdf;
      const int offset = 0;
      init(chan, eof_pdf, offset);
      }
protected:
   // Interface with derived classes
   void advance() const;
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1vd_t& app, array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
         array1d_t& eof_post, const libbase::size_type<libbase::vector> offset);
   // Internal methods
   void demodulate_wrapper(const channel<bool>& chan, const array1b_t& rx,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
         array1d_t& eof_post, const int offset);
private:
   /*! \name Internal functions */
   static void normalize(const array1r_t& in, array1d_t& out);
   // @}
public:
   /*! \name Constructors / Destructors */
   explicit dminner2(const int n = 2, const int k = 1) :
      dminner<real> (n, k)
      {
      }
   dminner2(const int n, const int k, const double th_inner,
         const double th_outer) :
      dminner<real> (n, k, th_inner, th_outer)
      {
      }
   // @}

   // Block modem operations
   // (necessary because inheriting methods from templated base)
   using Interface::modulate;
   using Interface::demodulate;

   // Block modem operations - streaming extensions
   void get_post_drift_pdf(array1vd_t& pdftable) const
      {
      // get the posterior channel drift pdf at codeword boundaries
      array1vr_t pdftable_r;
      fba.get_drift_pdf(pdftable_r);
      Base::normalize_results(pdftable_r, pdftable);
      }
   array1i_t get_boundaries(void) const
      {
      // inherit block size from last modulation step
      const int n = Base::n;
      const int N = Base::marker.size();
      // construct list of codeword boundary positions
      array1i_t postable(N + 1);
      for (int i = 0; i <= N; i++)
         postable(i) = i * n;
      return postable;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dminner2)
};

} // end namespace

#endif
