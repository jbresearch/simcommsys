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

#ifndef __codec_softout_mapped_h
#define __codec_softout_mapped_h

#include "config.h"
#include "mapper/map_straight.h"
#include "symbol_converter.h"
#include "vectorutils.h"
#include "hard_decision.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show mapping block sizes
// 3 - Show soft-output for encoded symbols
// 4 - Show intermediate encoded output
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Channel Codec with Soft Output and same Input/Output Symbol Space.
 * \author  Johann Briffa
 */

template <class base_codec_softout, class dbl = double>
class codec_softout_mapped : public base_codec_softout {
private:
   // Shorthand for class hierarchy
   typedef base_codec_softout Base;
   typedef codec_softout_mapped<base_codec_softout, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   /*! \name Internal functions */
   template <class D> void init(mapper<libbase::vector, D>& map) const;
   // @}
protected:
   // Interface with derived classes
   void do_encode(const array1i_t& source, array1i_t& encoded);
   void do_init_decoder(const array1vd_t& ptable);
   void do_init_decoder(const array1vd_t& ptable, const array1vd_t& app);
public:
   /*! \name Constructors / Destructors */
   ~codec_softout_mapped()
      {
      }
   // @}

   // Codec operations
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // Inherit sizes
      const int N = Base::output_block_size();
      const int n = Base::num_outputs();
      const int k = Base::num_inputs();
      return libbase::size_type<libbase::vector>(
            int(round(N * log(n) / log(k))));
      }
   int num_outputs() const
      {
      return Base::num_inputs();
      }

   // Description
   std::string description() const
      {
      return "Flattened " + Base::description();
      }
};

// Initialization

template <class base_codec_softout, class dbl>
template <class D>
void codec_softout_mapped<base_codec_softout, dbl>::init(mapper<
      libbase::vector, D>& map) const
   {
   // Set up mapper
   const int N = Base::num_outputs(); // codec output alphabet
   const int M = This::num_outputs(); // blockmodem input alphabet
   map.set_parameters(N, M);
   map.set_blocksize(Base::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG: mapper setup from "
   << map.input_block_size() << "×" << N << " to "
   << map.output_block_size() << "×" << M << " symbols" << std::endl;
#endif
   }

// Interface with derived classes

template <class base_codec_softout, class dbl>
void codec_softout_mapped<base_codec_softout, dbl>::do_encode(
      const array1i_t& source, array1i_t& encoded)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Encode to a temporary space and convert
   array1i_t encwide;
   Base::encode(source, encwide);
   map.transform(encwide, encoded);
#if DEBUG>=4
   std::cerr << "Source:" << std::endl;
   source.serialize(std::cerr, '\n');
   std::cerr << "Encoded (wide):" << std::endl;
   encwide.serialize(std::cerr, '\n');
   std::cerr << "Encoded (flat):" << std::endl;
   encoded.serialize(std::cerr, '\n');
#endif
   }

template <class base_codec_softout, class dbl>
void codec_softout_mapped<base_codec_softout, dbl>::do_init_decoder(
      const array1vd_t& ptable)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Convert to a temporary space and translate
   array1vd_t ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::init_decoder(ptable_flat);
   }

template <class base_codec_softout, class dbl>
void codec_softout_mapped<base_codec_softout, dbl>::do_init_decoder(
      const array1vd_t& ptable, const array1vd_t& app)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Convert to a temporary space and translate
   array1vd_t ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::init_decoder(ptable_flat, app);
   }

// Codec operations

template <class base_codec_softout, class dbl>
void codec_softout_mapped<base_codec_softout, dbl>::softdecode(
      array1vd_t& ri, array1vd_t& ro)
   {
   // Decode to a temporary space
   array1vd_t ro_wide;
   Base::softdecode(ri, ro_wide);
#if DEBUG>=3
   array1i_t dec;
   hard_decision<libbase::vector, dbl, int> functor;
   functor(ro_wide,dec);
   libbase::trace << "DEBUG (csf): ro_wide = ";
   dec.serialize(libbase::trace, ' ');
#endif
   // Allocate space for results
   libbase::allocate(ro, This::output_block_size(), This::num_outputs());
   // Initialize
   ro = 0.0;
   // Convert
   const int N = Base::num_outputs(); // # enc outputs
   const int M = This::num_outputs(); // # mod symbols
   const int s1 = libbase::symbol_converter<>::get_rate(M, N);
   for (int t = 0; t < ro_wide.size(); t++)
      for (int x = 0; x < ro_wide(t).size(); x++)
         for (int i = 0, thisx = x; i < s1; i++, thisx /= M)
            ro(t * s1 + i)(thisx % M) += ro_wide(t)(x);
#if DEBUG>=3
   functor(ro,dec);
   libbase::trace << "DEBUG (csf): ro = ";
   dec.serialize(libbase::trace, ' ');
#endif
   }

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
