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

#ifndef __mapper_h
#define __mapper_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "serializer.h"
#include "random.h"
#include "blockprocess.h"
#include "instrumented.h"
#include <iostream>
#include <string>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show input/output for mapping process
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Mapper Interface.
 * \author  Johann Briffa
 *
 * This class defines the interface for mapper classes. It is used within
 * commsys as a layer between codec and blockmodem. It transforms the codec
 * output sequence to a blockmodem input sequence, and inverts the
 * transformation for the symbol probabilities from the blockmodem. The codec
 * reads these to set up its receiver (ie the probabilities of what was
 * received).
 *
 * For full-system iteration, an additional method transforms the (extrinsic)
 * probabilities from codec output to modem input (where these are used as
 * priors).
 *
 * \tparam C Container class for codec/modem blocks
 * \tparam dbl Floating-point type for probability tables
 */

template <template <class > class C = libbase::vector, class dbl = double>
class mapper : public instrumented,
      public blockprocess,
      public libbase::serializable {
private:
   // Shorthand for class hierarchy
   typedef mapper<C, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

protected:
   /*! \name User-defined parameters */
   int q; //!< Alphabet size for encoder output
   int M; //!< Alphabet size for blockmodem input
   libbase::size_type<C> size; //!< Input block size in symbols
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Setup function, called from set_parameters and set_blocksize
   virtual void setup()
      {
      }
   //! \copydoc transform()
   virtual void dotransform(const C<int>& in, C<int>& out) const = 0;
   //! \copydoc transform()
   virtual void dotransform(const C<array1d_t>& pin, C<array1d_t>& pout) const = 0;
   //! \copydoc inverse()
   virtual void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~mapper()
      {
      }
   // @}

   /*! \name Vector mapper operations */
   /*!
    * \brief Transform a encoder output sequence to blockmodem input (N->M)
    * \param[in]  in    Encoder output sequence
    * \param[out] out   Modulator input sequence
    */
   void transform(const C<int>& in, C<int>& out) const
      {
      advance_always();
      dotransform(in, out);
#if DEBUG>=2
      std::cerr << "DEBUG (mapper): transform in = " << in;
      std::cerr << "DEBUG (mapper): transform out = " << out;
#endif
      }
   /*!
    * \brief Transform the encoder output posteriors to blockmodem priors (N->M)
    * \param[in]  pin   Table of output posterior likelihoods from decoder
    * \param[out] pout  Table of prior likelihoods for blockmodem
    *
    * \note p(i,d) is the a posteriori probability of symbol 'd' at time 'i'
    *
    * \note An empty input matrix is handled as a special condition
    *
    * \note Since this method is always called as part of an iterative process,
    * there is never a reason to advance or mark as dirty.
    */
   void transform(const C<array1d_t>& pin, C<array1d_t>& pout) const
      {
      if (pin.size() == 0)
         pout = pin;
      else
         dotransform(pin, pout);
#if DEBUG>=2
      std::cerr << "DEBUG (mapper): transform pin = " << pin;
      std::cerr << "DEBUG (mapper): transform pout = " << pout;
#endif
      }
   /*!
    * \brief Inverse-transform the blockmodem receiver probabilities to decoder
    * input (M->N)
    * \param[in]  pin   Table of likelihoods from demodulator
    * \param[out] pout  Table of likelihoods for decoder
    *
    * \note An empty input matrix is handled as a special condition
    *
    * \note p(i,d) is the a posteriori probability of symbol 'd' at time 'i'
    */
   void inverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
      {
      advance_if_dirty();
      if (pin.size() == 0)
         pout = pin;
      else
         doinverse(pin, pout);
      mark_as_dirty();
#if DEBUG>=2
      std::cerr << "DEBUG (mapper): inverse pin = " << pin;
      std::cerr << "DEBUG (mapper): inverse pout = " << pout;
#endif
      }
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   /*!
    * \brief Sets input and output alphabet sizes
    * \param[in]  q  Alphabet size for encoder output
    * \param[in]  M  Alphabet size for blockmodem input
    */
   void set_parameters(const int q, const int M)
      {
      this->q = q;
      this->M = M;
      setup();
      }
   //! Sets input block size (as at encoder output)
   void set_blocksize(libbase::size_type<C> size)
      {
      this->size = size;
      setup();
      }
   // @}

   /*! \name Informative functions */
   //! Overall mapper rate
   virtual double rate() const
      {
      return 1;
      }
   //! Gets input block size
   libbase::size_type<C> input_block_size() const
      {
      return size;
      }
   //! Gets output block size
   virtual libbase::size_type<C> output_block_size() const
      {
      return size;
      }
   // @}

   /*! \name Description */
   //! Object description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(mapper)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
