/*!
 * \file
 * $Id: uncoded.h 9909 2013-09-23 08:43:23Z jabriffa $
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

#ifndef __conv_h
#define __conv_h

#include <string>

#include "config.h"

#include "codec_softout.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Uncoded transmission.
 * \author  Johann Briffa
 * $Id: uncoded.h 9909 2013-09-23 08:43:23Z jabriffa $
 *
 * This class represents the simplest possible encoding, where the output is
 * simply a copy of the input. Equivalently, at the receiving end, the
 * decoder soft-output is simply a copy of its soft-input.
 */

template <class dbl = double>
class conv : public codec_softout<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef conv<dbl> This;
   typedef codec_softout<libbase::vector, dbl> Base;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   /*! \name User-specified parameters */
   int q; //!< Alphabet size (input and output)
   int N; //!< Length of input/output sequence
   int inp_bits; //number of input bits
   int out_bits; //number of output bits
   std::string temp;
   // @}
   /*! \name Computed parameters */
   array1vd_t rp; //!< Intrinsic source statistics
   array1vd_t R; //!< Intrinsic output statistics
   // @}
protected:
   // Internal codec_softout operations
   void resetpriors();
   void setpriors(const array1vd_t& ptable);
   void setreceiver(const array1vd_t& ptable);
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   conv(int q=2, int N=1) :
      q(q), N(N)
      {
      }
   // @}

   // Codec operations
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      return libbase::size_type<libbase::vector>(N);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(N);
      }
   int num_inputs() const
      {
      return q;
      }
   int num_outputs() const
      {
      return q;
      }
   int tail_length() const
      {
      return 0;
      }
   int num_iter() const
      {
      return 1;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(conv)
};

} // end namespace

#endif

