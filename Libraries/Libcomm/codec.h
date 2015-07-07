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

#ifndef __codec_h
#define __codec_h

#include "config.h"
#include "matrix.h"
#include "vector.h"
#include "serializer.h"
#include "random.h"
#include "instrumented.h"
#include "blockprocess.h"
#include "cputimer.h"
#include <string>

namespace libcomm {

/*!
 * \brief   Channel Codec Base.
 * \author  Johann Briffa
 *
 * \todo Original model assumed one symbol per timestep; this is no longer the
 * case. Confirm there are no lingering assumptions of this.
 *
 * \todo Remove tail_length() as tailing should be handled internally
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec : public instrumented,
      public blockprocess,
      public libbase::serializable {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

protected:
   /*! \name Interface with derived classes */
   //! \copydoc encode()
   virtual void do_encode(const C<int>& source, C<int>& encoded) = 0;
   //! \copydoc init_decoder()
   virtual void do_init_decoder(const C<array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~codec()
      {
      }
   // @}

   /*! \name Codec operations */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   /*!
    * \brief Encoding process
    * \param[in] source Sequence of source symbols
    * \param[out] encoded Sequence of output (encoded) symbols
    */
   void encode(const C<int>& source, C<int>& encoded)
      {
      libbase::cputimer t("t_encode");
      advance_always();
      do_encode(source, encoded);
      add_timer(t);
      }
   /*!
    * \brief Receiver translation process
    * \param[in] ptable Likelihoods of each possible encoded symbol at every index
    *
    * This function initializes the decoder with the probability tables for
    * each encoded symbol as received from the blockmodem.
    * This function should be called before the first decode iteration
    * for each block.
    */
   void init_decoder(const C<array1d_t>& ptable)
      {
      libbase::cputimer t("t_init_decoder");
      advance_if_dirty();
      do_init_decoder(ptable);
      mark_as_dirty();
      add_timer(t);
      }
   /*!
    * \brief Decoding process
    * \param[out] decoded Most likely sequence of information symbols
    *
    * \note Observe that this output necessarily constitutes a hard decision.
    *
    * \note Each call to decode will perform a single iteration (with respect
    * to num_iter).
    */
   virtual void decode(C<int>& decoded) = 0;
   // @}

   /*! \name Codec information functions - fundamental */
   //! Input block size in symbols
   virtual libbase::size_type<C> input_block_size() const = 0;
   //! Output block size in symbols
   virtual libbase::size_type<C> output_block_size() const = 0;
   //! Input alphabet size (number of valid symbols)
   virtual int num_inputs() const = 0;
   //! Output alphabet size (number of valid symbols)
   virtual int num_outputs() const = 0;
   //! Length of tail in timesteps
   virtual int tail_length() const = 0;
   //! Number of iterations per decoding cycle
   virtual int num_iter() const = 0;
   // @}

   /*! \name Codec information functions - derived */
   //! Equivalent length of information sequence in bits
   double input_bits() const
      {
      return log2(num_inputs()) * input_block_size();
      }
   //! Equivalent length of output sequence in bits
   double output_bits() const
      {
      return log2(num_outputs()) * output_block_size();
      }
   //! Overall code rate
   double rate() const
      {
      return input_bits() / output_bits();
      }
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(codec)
};

} // end namespace

#endif

