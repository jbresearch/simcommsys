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

#ifndef __codec_concatenated_h
#define __codec_concatenated_h

#include "config.h"
#include "codec_softout.h"
#include "mapper.h"
#include <list>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Channel Codec representing a concatenation of codecs.
 * \author  Johann Briffa
 */

template <template <class > class C = libbase::vector, class dbl = double>
class codec_concatenated : public codec_softout<C, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef std::list<codec_softout<C, dbl> *> codec_list_t;
   typedef std::list<mapper<C, dbl> *> mapper_list_t;
   // @}
private:
   /*! \name Internal representation */
   codec_list_t codec_list;
   mapper_list_t mapper_list;
   // @}
protected:
   /*! \name Internal operations */
   //! Invariance test
   void test_invariant() const
      {
      // The tests below assume a properly set up and usable system
      assert(codec_list.size() >= 1);
      assert(mapper_list.size() == codec_list.size() - 1);
      }
   void free()
      {
      // Destroy codecs
      for(typename codec_list_t::iterator it = codec_list.begin(); it != codec_list.end(); it++)
         delete *it;
      codec_list.clear();
      // Destroy mappers
      for(typename mapper_list_t::iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         delete *it;
      mapper_list.clear();
      }
   void init();
   // @}
   // Interface with derived classes
   void do_encode(const C<int>& source, C<int>& encoded);
   void do_init_decoder(const C<array1d_t>& ptable)
      {
      test_invariant();
      // Initialize the first codec in line (in reverse order)
      codec_list.back()->init_decoder(ptable);
      }
   void do_init_decoder(const C<array1d_t>& ptable, const C<array1d_t>& app)
      {
      test_invariant();
      // Initialize the first codec in line (in reverse order)
      codec_list.back()->init_decoder(ptable, app);
      }
public:
   /*! \name Constructors / Destructors */
   ~codec_concatenated()
      {
      free();
      }
   // @}

   // Codec operations
   void seedfrom(libbase::random& r)
      {
      test_invariant();
      // Seed codecs
      for(typename codec_list_t::iterator it = codec_list.begin(); it != codec_list.end(); it++)
         (*it)->seedfrom(r);
      // Seed mappers
      for(typename mapper_list_t::iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         (*it)->seedfrom(r);
      }
   void softdecode(C<array1d_t>& ri);
   void softdecode(C<array1d_t>& ri, C<array1d_t>& ro);

   // Codec information functions - fundamental
   libbase::size_type<C> input_block_size() const
      {
      test_invariant();
      // Input size is same as that for first codec
      return codec_list.front()->input_block_size();
      }
   libbase::size_type<C> output_block_size() const
      {
      test_invariant();
      // Output size is same as that for last codec
      return codec_list.back()->output_block_size();
      }
   int num_inputs() const
      {
      test_invariant();
      // Input size is same as that for first codec
      return codec_list.front()->num_inputs();
      }
   int num_outputs() const
      {
      test_invariant();
      // Output size is same as that for last codec
      return codec_list.back()->num_outputs();
      }
   int tail_length() const
      {
      test_invariant();
      // Tail length is same as that for first codec
      return codec_list.front()->tail_length();
      }
   int num_iter() const
      {
      return 1; // TODO: implement iterative decoding for sequence
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(codec_concatenated)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
