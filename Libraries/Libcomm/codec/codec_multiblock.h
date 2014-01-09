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

#ifndef __codec_multiblock_h
#define __codec_multiblock_h

#include "config.h"
#include "codec_softout.h"

#include "boost/shared_ptr.hpp"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Channel Codec aggregating multiple blocks of underlying codec.
 * \author  Johann Briffa
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_multiblock : public codec_softout<C, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   boost::shared_ptr<codec_softout<C, dbl> > cdc; //!< Underlying codec
   int N; //! Number of blocks to aggregate
   // @}
   /*! \name Internally-used objects */
   boost::shared_ptr<codec_softout<C, dbl> > cdc_enc; //!< Copy of codec object for encoder operations
   C<array1d_t> ptable; //!< Copy of channel probabilities, to be segmented and used
   C<array1d_t> app; //!< Copy of prior probabilities, to be segmented and used
   // @}
protected:
   /*! \name Internal operations */
   //! Invariance test
   void test_invariant() const
      {
      // The tests below assume a properly set up and usable system
      assert(cdc);
      assert(N >= 1);
      }
   // @}
   // Interface with derived classes
   void do_encode(const C<int>& source, C<int>& encoded);
   void do_init_decoder(const C<array1d_t>& ptable)
      {
      test_invariant();
      this->ptable = ptable;
      this->app.init(0);
      }
   void do_init_decoder(const C<array1d_t>& ptable, const C<array1d_t>& app)
      {
      test_invariant();
      this->ptable = ptable;
      this->app = app;
      }
public:
   // Codec operations
   void seedfrom(libbase::random& r)
      {
      test_invariant();
      // Seed codec
      cdc->seedfrom(r);
      // Make a copy of codec object for encoder operations
      cdc_enc.reset(dynamic_cast<codec_softout<C, dbl> *>(cdc->clone()));
      }
   void softdecode(C<array1d_t>& ri);
   void softdecode(C<array1d_t>& ri, C<array1d_t>& ro);

   // Codec information functions - fundamental
   libbase::size_type<C> input_block_size() const
      {
      test_invariant();
      // Input size is a multple of that for codec
      const int tau = cdc->input_block_size() * N;
      return libbase::size_type<C>(tau);
      }
   libbase::size_type<C> output_block_size() const
      {
      test_invariant();
      // Output size is a multple of that for codec
      const int tau = cdc->output_block_size() * N;
      return libbase::size_type<C>(tau);
      }
   int num_inputs() const
      {
      test_invariant();
      // Input size is same as that for codec
      return cdc->num_inputs();
      }
   int num_outputs() const
      {
      test_invariant();
      // Output size is same as that for codec
      return cdc->num_outputs();
      }
   int tail_length() const
      {
      test_invariant();
      // Tail length is a multple of that for codec
      return cdc->tail_length() * N;
      }
   int num_iter() const
      {
      return 1;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(codec_multiblock)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
