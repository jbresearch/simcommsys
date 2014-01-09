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

#ifndef __mapcc_h
#define __mapcc_h

#include "config.h"

#include "codec_softout.h"
#include "fsm.h"
#include "safe_bcjr.h"
#include "itfunc.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>

namespace libcomm {

/*!
 * \brief   Maximum A-Posteriori Decoder.
 * \author  Johann Briffa
 *
 * \todo Update decoding process for changes in FSM model.
 */

template <class real, class dbl = double>
class mapcc : public codec_softout<libbase::vector, dbl> , private safe_bcjr<
      real, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapcc<real, dbl> This;
   typedef codec_softout<libbase::vector, dbl> Base;
   typedef safe_bcjr<real, dbl> BCJR;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::matrix<dbl> array2d_t;
   // @}
private:
   /*! \name User-defined parameters */
   fsm *encoder;
   int tau; //!< Sequence length in timesteps (including tail, if any)
   bool endatzero; //!< True for terminated trellis
   bool circular; //!< True for circular trellis
   // @}
   /*! \name Internal object representation */
   double rate;
   array2d_t R; //!< BCJR a-priori receiver statistics
   array2d_t app; //!< BCJR a-priori input statistics
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   void reset();
   // @}
   // Internal codec operations
   void resetpriors();
   void setpriors(const array1vd_t& ptable);
   void setreceiver(const array1vd_t& ptable);
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable)
      {
      setreceiver(ptable);
      resetpriors();
      }
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app)
      {
      setreceiver(ptable);
      setpriors(app);
      }
public:
   /*! \name Law of the Big Three */
   //! Destructor
   virtual ~mapcc()
      {
      free();
      }
   //! Copy constructor
   mapcc(const mapcc<real, dbl>& x) :
         tau(x.tau), endatzero(x.endatzero), circular(x.circular)
      {
      if (x.encoder)
         {
         encoder = dynamic_cast<fsm*>(x.encoder->clone());
         init();
         }
      else
         encoder = NULL;
      }
   //! Copy assignment operator
   mapcc<real, dbl>& operator=(const mapcc<real, dbl>& x)
      {
      tau = x.tau;
      endatzero = x.endatzero;
      circular = x.circular;
      if (x.encoder)
         {
         encoder = dynamic_cast<fsm*>(x.encoder->clone());
         init();
         }
      else
         encoder = NULL;
      return *this;
      }
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   mapcc() :
      encoder(NULL)
      {
      }
   //! Principal constructor
   mapcc(const fsm& encoder, const int tau, const bool endatzero,
         const bool circular) :
         encoder(dynamic_cast<fsm*>(encoder.clone())), tau(tau), endatzero(
               endatzero), circular(circular)
      {
      init();
      }
   // @}

   // Codec operations
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      assertalways(encoder);
      const int nu = This::tail_length();
      const int k = encoder->num_inputs();
      const int result = (tau - nu) * k;
      return libbase::size_type<libbase::vector>(result);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      assertalways(encoder);
      const int n = encoder->num_outputs();
      const int result = tau * n;
      return libbase::size_type<libbase::vector>(result);
      }
   int num_inputs() const
      {
      assertalways(encoder);
      return encoder->num_symbols();
      }
   int num_outputs() const
      {
      assertalways(encoder);
      return encoder->num_symbols();
      }
   int tail_length() const
      {
      assertalways(encoder);
      return endatzero ? encoder->mem_order() : 0;
      }
   int num_iter() const
      {
      return 1;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(mapcc)
};

} // end namespace

#endif

