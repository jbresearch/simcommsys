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

#ifndef __blockembedder_h
#define __blockembedder_h

#include "vector.h"
#include "matrix.h"
#include "serializer.h"
#include "blockprocess.h"
#include "channel.h"

namespace libcomm {

/*!
 * \brief   Blockwise Data Embedder/Extractor Common Interface.
 * \author  Johann Briffa
 *
 * Class defines common interface for blockembedder classes.
 */

template <class S, template <class > class C = libbase::vector,
      class dbl = double>
class basic_blockembedder : public blockprocess {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}

private:
   /*! \name User-defined parameters */
   libbase::size_type<C> size; //!< Input block size in symbols
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Setup function, called from set_blocksize() in base class
   virtual void setup()
      {
      }
   //! Validates block size, called from modulate() and demodulate()
   virtual void test_invariant() const
      {
      assert(size > 0);
      }
   //! \copydoc embed()
   virtual void doembed(const int N, const C<int>& data, const C<S>& host,
         C<S>& stego) = 0;
   //! \copydoc extract()
   virtual void doextract(const channel<S, C>& chan, const C<S>& rx, C<
         array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_blockembedder()
      {
      }
   // @}

   // Atomic embedder operations
   // (necessary because overloaded methods hide those in templated base)
   //using embedder<S>::embed;
   //using embedder<S>::extract;

   /*! \name Block embedder operations */
   /*!
    * \brief Embed a sequence of symbols
    * \param[in]  N        The number of possible values of each encoded element
    * \param[in]  data     Sequence of data values to be embedded
    * \param[in]  host     Sequence of host values into which to embed data
    * \param[out] stego    Sequence of stego-values corresponding to the given input
    *
    * \todo Remove parameter N, replacing 'int' type for data vector with
    * something that also encodes the number of symbols in the alphabet.
    *
    * \note This function is non-const, to support time-variant embedding
    * schemes such as (key-dependent) SSIS.
    */
   void embed(const int N, const C<int>& data, const C<S>& host, C<S>& stego);
   /*!
    * \brief Extract a sequence of symbols
    * \param[in]  chan     The channel model (used to obtain likelihoods)
    * \param[in]  rx       Sequence of received (possibly corrupted) stego-values
    * \param[out] ptable   Table of likelihoods of possible transmitted symbols
    *
    * \note \c ptable(i)(d) \c is the a posteriori probability of having
    * transmitted symbol 'd' at time 'i'
    *
    * \note This function is non-const, to support time-variant modulation
    * schemes such as (key-dependent) SSIS.
    */
   void
   extract(const channel<S, C>& chan, const C<S>& rx, C<array1d_t>& ptable);
   // @}

   /*! \name Setup functions */
   //! Sets input block size
   void set_blocksize(libbase::size_type<C> size)
      {
      assert(size > 0);
      this->size = size;
      this->setup();
      }
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   // @}

   /*! \name Informative functions */
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
   //! Symbol alphabet size at input
   virtual int num_symbols() const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 * \brief   Blockwise Data Embedder/Extractor Base.
 * \author  Johann Briffa
 *
 * Class defines base interface for blockembedder classes.
 */

template <class S, template <class > class C = libbase::vector,
      class dbl = double>
class blockembedder : public basic_blockembedder<S, C, dbl> ,
      public libbase::serializable {
public:
   // Serialization Support
DECLARE_BASE_SERIALIZER(blockembedder)
};

} // end namespace

#endif
