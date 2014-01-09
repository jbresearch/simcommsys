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

#ifndef __blockmodem_h
#define __blockmodem_h

#include "modem.h"
#include "vector.h"
#include "matrix.h"
#include "channel.h"
#include "blockprocess.h"
#include "instrumented.h"
#include "cputimer.h"

namespace libcomm {

/*!
 * \brief   Blockwise Modulator Common Interface.
 * \author  Johann Briffa
 *
 * Class defines common interface for blockmodem classes.
 */

template <class S, template <class > class C = libbase::vector,
      class dbl = double>
class basic_blockmodem : public instrumented,
      public modem<S> ,
      public blockprocess {
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
   //! \copydoc modulate()
   virtual void domodulate(const int N, const C<int>& encoded, C<S>& tx) = 0;
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S, C>& chan, const C<S>& rx, C<
         array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_blockmodem()
      {
      }
   // @}

   // Atomic modem operations
   // (necessary because overloaded methods hide those in templated base)
   using modem<S>::modulate;
   using modem<S>::demodulate;

   /*! \name Block modem operations */
   /*!
    * \brief Modulate a sequence of time-steps
    * \param[in]  N        The number of possible values of each encoded element
    * \param[in]  encoded  Sequence of values to be modulated
    * \param[out] tx       Sequence of symbols corresponding to the given input
    *
    * \todo Remove parameter N, replacing 'int' type for encoded vector with
    * something that also encodes the number of symbols in the alphabet.
    *
    * \note This function is non-const, to support time-variant modulation
    * schemes such as DM inner codes.
    */
   void modulate(const int N, const C<int>& encoded, C<S>& tx)
      {
      test_invariant();
      libbase::cputimer t("t_modulate");
      advance_always();
      domodulate(N, encoded, tx);
      add_timer(t);
      }
   /*!
    * \brief Demodulate a sequence of time-steps
    * \param[in]  chan     The channel model (used to obtain likelihoods)
    * \param[in]  rx       Sequence of received symbols
    * \param[out] ptable   Table of likelihoods of possible transmitted symbols
    *
    * \note \c ptable(i)(d) \c is the a posteriori probability of having
    * transmitted symbol 'd' at time 'i'
    *
    * \note This function is non-const, to support time-variant modulation
    * schemes such as DM inner codes.
    */
   void demodulate(const channel<S, C>& chan, const C<S>& rx,
         C<array1d_t>& ptable)
      {
      test_invariant();
      libbase::cputimer t("t_demodulate");
      advance_if_dirty();
      dodemodulate(chan, rx, ptable);
      mark_as_dirty();
      add_timer(t);
      }
   // @}

   /*! \name Setup functions */
   //! Sets input block size
   void set_blocksize(libbase::size_type<C> size)
      {
      assert(size > 0);
      this->size = size;
      this->setup();
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
   // @}
};

/*!
 * \brief   Blockwise Modulator Base.
 * \author  Johann Briffa
 *
 * Class defines base interface for blockmodem classes.
 */

template <class S, template <class > class C = libbase::vector,
      class dbl = double>
class blockmodem : public basic_blockmodem<S, C, dbl> ,
      public libbase::serializable {
public:
   //! Virtual destructor
   virtual ~blockmodem()
      {
      }
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(blockmodem)
};

} // end namespace

#endif
