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

#ifndef __commsys_stream_h
#define __commsys_stream_h

#include "commsys.h"
#include "modem/stream_modulator.h"
#include "channel_stream.h"

namespace libcomm {

/*!
 * \brief   Communication System supporting stream synchronization.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Communication system that supports stream synchronization. Consequently,
 * the reception process has an enhanced interface. This:
 * 1) allows the user to supply a received sequence that overlaps with the
 *    previous and next frames,
 * 2) allows the user to supply prior information on where the frame is
 *    likely to begin/end, and
 * 3) allows the user to extract posterior information on where the frame is
 *    likely to begin/end.
 *
 * This class requires the underlying modem and channel to support stream
 * operations. Access methods are also provided.
 */

template <class S, template <class > class C = libbase::vector>
class commsys_stream : public commsys<S, C> {
private:
   // Shorthand for class hierarchy
   typedef commsys_stream<S, C> This;
   typedef commsys<S, C> Base;

public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}

private:
   /*! \name Internally-used objects */
   C<double> sof_post, eof_post;
   // @}

public:
   /*! \name Stream Utilities */
   static libbase::size_type<C> estimate_drift(const C<double>& pdf,
         const libbase::size_type<C> offset)
      {
      const int drift = libbase::index_of_max(pdf) - offset;
      return libbase::size_type<C>(drift);
      }
   static C<double> centralize_pdf(const C<double>& pdf, const libbase::size_type<C> drift)
      {
      C<double> newpdf(pdf.size());
      newpdf = 0;
      const int sh_a = std::max(0, -drift);
      const int sh_b = std::max(0, int(drift));
      const int sh_n = pdf.size() - abs(drift);
      newpdf.segment(sh_a, sh_n) = pdf.extract(sh_b, sh_n);
      return newpdf;
      }
   // @}

   /*! \name Communication System Setup - Stream Extensions */
   //! Get modulation scheme in stream mode
   stream_modulator<S, C>& getmodem_stream() const
      {
      return dynamic_cast<stream_modulator<S, C>&> (*this->mdm);
      }
   //! Get receiver channel model in stream mode
   channel_stream<S>& getrxchan_stream() const
      {
      return dynamic_cast<channel_stream<S>&> (*this->rxchan);
      }
   //! Get transmit channel model in stream mode
   channel_stream<S>& gettxchan_stream() const
      {
      return dynamic_cast<channel_stream<S>&> (*this->txchan);
      }
   // @}

   /*! \name Communication System Interface - Stream Extensions */
   void compute_priors(const C<double>& eof_post, C<double>& sof_prior, C<
         double>& eof_prior, libbase::size_type<C>& offset) const;
   void receive_path(const C<S>& received, const C<double>& sof_prior, const C<
         double>& eof_prior, const libbase::size_type<C> offset);
   const C<double>& get_sof_post() const
      {
      return sof_post;
      }
   const C<double>& get_eof_post() const
      {
      return eof_post;
      }
   // @}

   // Description
   std::string description() const;
   // Serialization Support
DECLARE_SERIALIZER(commsys_stream)
};

} // end namespace

#endif
