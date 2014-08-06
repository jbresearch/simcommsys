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

#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "randgen.h"
#include "codec.h"
#include "mapper.h"
#include "blockmodem.h"
#include "channel.h"
#include "serializer.h"
#include "instrumented.h"

namespace libcomm {

/*!
 * \brief   Common Base for Communication System.
 * \author  Johann Briffa
 *
 * General templated commsys.
 * - Integrates functionality of binary variant.
 * - Explicit instantiations for bool and gf types are present.
 *
 * \todo Consider removing subcomponent getters, enforcing calls through this
 * interface
 */

template <class S, template <class > class C = libbase::vector>
class basic_commsys : public instrumented {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}

protected:
   /*! \name Bound objects */
   codec<C> *cdc; //!< Error-control codec
   mapper<C> *map; //!< Symbol-mapper (encoded output to transmitted symbols)
   blockmodem<S, C> *mdm; //!< Modulation scheme
   channel<S, C> *txchan; //!< Channel model - transmitter side
   channel<S, C> *rxchan; //!< Channel model - receiver side
   bool singlechannel; //!< Flag indicating RX = TX channel
   // @}
#ifndef NDEBUG
   bool lastframecorrect;
   C<int> lastsource;
#endif
protected:
   /*! \name Setup functions */
   void init();
   void clear();
   void free();
   // @}
public:
   /*! \name Constructors / Destructors */
   basic_commsys(const basic_commsys<S, C>& c);
   basic_commsys()
      {
      clear();
      }
   virtual ~basic_commsys()
      {
      free();
      }
   // @}

   /*! \name Communication System Setup */
   virtual void seedfrom(libbase::random& r);
   //! Get error-control codec
   codec<C> *getcodec() const
      {
      return cdc;
      }
   //! Get symbol mapper
   mapper<C> *getmapper() const
      {
      return map;
      }
   //! Get modulation scheme
   blockmodem<S, C> *getmodem() const
      {
      return mdm;
      }
   //! Get channel model - transmitter side
   channel<S, C> *gettxchan() const
      {
      return txchan;
      }
   //! Get channel model - receiver side
   channel<S, C> *getrxchan() const
      {
      return rxchan;
      }
   // @}

   /*! \name Communication System Interface */
   //! Perform complete encode path
   virtual C<S> encode_path(const C<int>& source);
   //! Perform channel transmission
   virtual C<S> transmit(const C<S>& transmitted);
   //! Perform complete receive path, except for final decoding
   virtual void receive_path(const C<S>& received);
   //! Perform after-demodulation receive path, except for final decoding
   virtual void softreceive_path(const C<array1d_t>& ptable_mapped);
   //! Perform a decoding iteration, with hard decision
   virtual void decode(C<int>& decoded);
   // @}

   /*! \name Informative functions */
   //! Number of iterations to perform
   virtual int num_iter() const
      {
      return cdc->num_iter();
      }
   //! Overall mapper rate
   double rate() const
      {
      return cdc->rate() * map->rate();
      }
   //! Input alphabet size (number of valid symbols)
   int num_inputs() const
      {
      return cdc->num_inputs();
      }
   //! Output alphabet size (number of valid symbols)
   int num_outputs() const
      {
      return mdm->num_symbols();
      }
   //! Input (ie. source/decoded) block size in symbols
   libbase::size_type<C> input_block_size() const
      {
      return cdc->input_block_size();
      }
   //! Output (ie. transmitted/received) block size in symbols
   libbase::size_type<C> output_block_size() const
      {
      return mdm->output_block_size();
      }
   // @}

   //! Clear list of timers
   void reset_timers()
      {
      // clear list of timers we're keeping
      instrumented::reset_timers();
      // clear list of timers for all components
      cdc->reset_timers();
      map->reset_timers();
      mdm->reset_timers();
      txchan->reset_timers();
      rxchan->reset_timers();
      }

   // Description
   virtual std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

/*!
 * \brief   General Communication System.
 * \author  Johann Briffa
 *
 * General templated commsys, directly derived from common base.
 */

template <class S, template <class > class C = libbase::vector>
class commsys : public basic_commsys<S, C> , public libbase::serializable {
public:
   // Serialization Support
DECLARE_BASE_SERIALIZER(commsys)
DECLARE_SERIALIZER(commsys)
};

/*!
 * \brief   Signal-Space Communication System.
 * \author  Johann Briffa
 *
 * This explicit specialization for sigspace channel contains objects and
 * functions remaining from the templated base, and is generally equivalent
 * to the old commsys class; anything that used to use 'commsys' can now use
 * this specialization.
 *
 * \note Support for puncturing has changed from its previous operation in
 * signal-space to the more general mapper layer.
 *
 * \note Serialization of puncturing system is implemented; the canonical
 * form this requires the addition of a 'false' flag at the end of the
 * stream to signal that there is no puncturing. In order not to break
 * current input files, the flag is assumed to be false (with no error)
 * if we have reached the end of the stream.
 */
template <template <class > class C>
class commsys<sigspace, C> : public basic_commsys<sigspace, C> ,
      public libbase::serializable {
protected:
   /*! \name Setup functions */
   void init();
   // @}
public:
   // Serialization Support
DECLARE_BASE_SERIALIZER(commsys)
DECLARE_SERIALIZER(commsys)
};

} // end namespace

#endif
