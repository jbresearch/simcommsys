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

#ifndef __codec_reshaped_h
#define __codec_reshaped_h

#include "config.h"
#include "codec.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show input/output of blocks being reshaped
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Channel Codec with matrix container from vector container.
 * \author  Johann Briffa
 */

template <class base_codec>
class codec_reshaped : public codec<libbase::matrix> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}
private:
   /*! \name Internal representation */
   base_codec base;
   // @}
protected:
   // Interface with derived classes
   void do_encode(const libbase::matrix<int>& source,
         libbase::matrix<int>& encoded)
      {
      libbase::vector<int> source_v = source.rowmajor();
      libbase::vector<int> encoded_v;
#if DEBUG>=2
      libbase::trace << "DEBUG (codec_reshaped): source = " << source;
      libbase::trace << "DEBUG (codec_reshaped): source_v = " << source_v;
#endif
      base.encode(source_v, encoded_v);
      encoded = encoded_v;
#if DEBUG>=2
      libbase::trace << "DEBUG (codec_reshaped): encoded = " << encoded;
      libbase::trace << "DEBUG (codec_reshaped): encoded_v = " << encoded_v;
#endif
      }
   void do_init_decoder(const libbase::matrix<array1d_t>& ptable)
      {
      libbase::vector<array1d_t> ptable_v = ptable.rowmajor();
      base.init_decoder(ptable_v);
      }
public:
   /*! \name Constructors / Destructors */
   ~codec_reshaped()
      {
      }
   // @}

   // Codec operations
   void seedfrom(libbase::random& r)
      {
      base.seedfrom(r);
      }
   void decode(libbase::matrix<int>& decoded)
      {
      libbase::vector<int> decoded_v;
      base.decode(decoded_v);
      decoded = decoded_v;
      }

   // Codec information functions - fundamental
   libbase::size_type<libbase::matrix> input_block_size() const
      {
      // Inherit sizes
      const int N = base.input_block_size();
      return libbase::size_type<libbase::matrix>(N, 1);
      }
   libbase::size_type<libbase::matrix> output_block_size() const
      {
      // Inherit sizes
      const int N = base.output_block_size();
      return libbase::size_type<libbase::matrix>(N, 1);
      }
   int num_inputs() const
      {
      return base.num_inputs();
      }
   int num_outputs() const
      {
      return base.num_outputs();
      }
   int tail_length() const
      {
      return base.tail_length();
      }
   int num_iter() const
      {
      return base.num_iter();
      }

   // Description
   std::string description() const
      {
      return "Reshaped " + base.description();
      }

   // Serialization Support
DECLARE_SERIALIZER(codec_reshaped)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
