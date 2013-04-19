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

#ifndef __uncoded_h
#define __uncoded_h

#include "config.h"

#include "codec_softout.h"
#include "fsm.h"
#include "serializer.h"
#include <cstdlib>
#include <cmath>

namespace libcomm {

/*!
 * \brief   Uncoded transmission.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
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
   //! FSM specifying input-output mapping; must have no memory
   fsm *encoder;
   int tau; //!< Number of time-steps
   // @}
   /*! \name Computed parameters */
   array1vd_t rp; //!< Intrinsic source statistics
   array1vd_t R; //!< Intrinsic output statistics
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   // @}
   /*! \name Codec information functions - internal */
   //! Number of encoder input symbols / timestep
   int enc_inputs() const
      {
      assert(encoder);
      return encoder->num_inputs();
      }
   //! Number of encoder output symbols / timestep
   int enc_outputs() const
      {
      assert(encoder);
      return encoder->num_outputs();
      }
   // @}
   // Internal codec operations
   void resetpriors();
   void setpriors(const array1vd_t& ptable);
   void setreceiver(const array1vd_t& ptable);
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   uncoded() :
      encoder(NULL)
      {
      }
   //! Copy constructor
   uncoded(const uncoded<dbl>& x) :
      encoder(dynamic_cast<fsm*> (x.encoder->clone())), tau(x.tau), rp(x.rp), R(x.R)
      {
      }
   uncoded(const fsm& encoder, const int tau);
   ~uncoded()
      {
      free();
      }
   // @}

   // Codec operations
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      const int k = enc_inputs();
      return libbase::size_type<libbase::vector>(tau * k);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      const int n = enc_outputs();
      return libbase::size_type<libbase::vector>(tau * n);
      }
   int num_inputs() const
      {
      return encoder->num_symbols();
      }
   int num_outputs() const
      {
      return encoder->num_symbols();
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

