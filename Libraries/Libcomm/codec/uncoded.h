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

#ifndef __uncoded_h
#define __uncoded_h

#include "config.h"

#include "codec_softout.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Uncoded transmission.
 * \author  Johann Briffa
 *
 * This class represents the simplest possible encoding, where the output is
 * simply a copy of the input. Equivalently, at the receiving end, the
 * decoder soft-output is simply a copy of its soft-input.
 */

template <class dbl = double>
class uncoded : public codec_softout<libbase::vector, dbl> {
private:
   // Shorthand for class hierarchy
   typedef uncoded<dbl> This;
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
   // @}
   /*! \name Computed parameters */
   array1vd_t R; //!< Stored statistics from receiver and priors
   // @}
protected:
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable);
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app);
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   uncoded(int q=2, int N=1) :
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
DECLARE_SERIALIZER(uncoded)
};

} // end namespace

#endif

