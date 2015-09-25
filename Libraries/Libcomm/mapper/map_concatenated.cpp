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

#include "map_concatenated.h"
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template <class > class C, class dbl>
void map_concatenated<C, dbl>::setup()
   {
   test_invariant();
   // iterators for mapper and interface to use
   typename mapper_list_t::iterator mapper_it = mapper_list.begin();
   typename interface_list_t::iterator interface_it = interface_list.begin();
   // set input alphabet and block size from codec
   int q = this->q;
   libbase::size_type<C> size = this->size;
   // repeat for all interfaces
   for(; interface_it != interface_list.end(); interface_it++, mapper_it++)
      {
      // set output alphabet from interface
      const int M = (*interface_it);
      // set up mapper with required parameters
      (*mapper_it)->set_parameters(q, M);
      (*mapper_it)->set_blocksize(size);
      // set next input alphabet and block size from this mapper's output
      q = M;
      size = (*mapper_it)->output_block_size();
      }
   // set up last mapper:
   // set output alphabet from modem
   const int M = this->M;
   // set up mapper with required parameters
   (*mapper_it)->set_parameters(q, M);
   (*mapper_it)->set_blocksize(size);
   // inherit output size from last mapper:
   osize = (*mapper_it)->output_block_size();
   test_invariant();
   }

template <template <class > class C, class dbl>
void map_concatenated<C, dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   test_invariant();
   // placeholders for intermediate results
   C<int> a = in;
   C<int> b;
   // pass through all mappers
   for (typename mapper_list_t::const_iterator mapper_it = mapper_list.begin();
         mapper_it != mapper_list.end(); mapper_it++)
      {
      (*mapper_it)->transform(a, b);
      // switch input/output for next step
      std::swap(a, b);
      }
   // copy result
   out = a;
   test_invariant();
   }

template <template <class > class C, class dbl>
void map_concatenated<C, dbl>::dotransform(const C<array1d_t>& pin,
      C<array1d_t>& pout) const
   {
   test_invariant();
   // placeholders for intermediate results
   C<array1d_t> a = pin;
   C<array1d_t> b;
   // pass through all mappers
   for (typename mapper_list_t::const_iterator mapper_it = mapper_list.begin();
         mapper_it != mapper_list.end(); mapper_it++)
      {
      (*mapper_it)->transform(a, b);
      // switch input/output for next step
      std::swap(a, b);
      }
   // copy result
   pout = a;
   test_invariant();
   }

template <template <class > class C, class dbl>
void map_concatenated<C, dbl>::doinverse(const C<array1d_t>& pin,
      C<array1d_t>& pout) const
   {
   test_invariant();
   // placeholders for intermediate results
   C<array1d_t> a = pin;
   C<array1d_t> b;
   // pass through all mappers (in reverse)
   for (typename mapper_list_t::const_reverse_iterator mapper_it = mapper_list.rbegin();
         mapper_it != mapper_list.rend(); mapper_it++)
      {
      (*mapper_it)->inverse(a, b);
      // switch input/output for next step
      std::swap(a, b);
      }
   // copy result
   pout = a;
   test_invariant();
   }

// Description

template <template <class > class C, class dbl>
std::string map_concatenated<C, dbl>::description() const
   {
   test_invariant();
   std::ostringstream sout;
   sout << "Concatenated mapper (" << mapper_list.size() << " mappers) [";
   // iterators for mapper and interface to use
   typename mapper_list_t::const_iterator mapper_it = mapper_list.begin();
   typename interface_list_t::const_iterator interface_it = interface_list.begin();
   // mappers description
   size_t i = 0;
   for(; mapper_it != mapper_list.end(); mapper_it++)
      {
      sout << "M" << ++i << ": " << (*mapper_it)->description();
      if (i < mapper_list.size())
         {
         sout << ", Interface size=" << (*interface_it) << ", ";
         interface_it++;
         }
      }
   sout << "]";
   return sout.str();
   }

// Serialization Support

template <template <class > class C, class dbl>
std::ostream& map_concatenated<C, dbl>::serialize(std::ostream& sout) const
   {
   test_invariant();
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Number of concatenated mappers" << std::endl;
   sout << mapper_list.size() << std::endl;
   // iterators for mapper and interface to use
   typename mapper_list_t::const_iterator mapper_it = mapper_list.begin();
   typename interface_list_t::const_iterator interface_it = interface_list.begin();
   // serialize mappers
   size_t i = 0;
   for(; mapper_it != mapper_list.end(); mapper_it++)
      {
      sout << "# Mapper " << ++i << std::endl;
      sout << *(mapper_it);
      if (i < mapper_list.size())
         {
         sout << "# Interface size" << std::endl;
         sout << (*interface_it) << std::endl;
         interface_it++;
         }
      }
   return sout;
   }

/*!
 * \version 1 Initial version
 */
template <template <class > class C, class dbl>
std::istream& map_concatenated<C, dbl>::serialize(std::istream& sin)
   {
   free();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // get number of mappers
   int N;
   sin >> libbase::eatcomments >> N >> libbase::verify;
   assertalways(N >= 1);
   // serialize mappers
   for(int i = 0; i < N; i++)
      {
      mapper<C, dbl> *this_mapper;
      sin >> libbase::eatcomments >> this_mapper >> libbase::verify;
      mapper_list.push_back(this_mapper);
      // serialize interface size if necessary
      if (i < N - 1)
         {
         int this_interface;
         sin >> libbase::eatcomments >> this_interface >> libbase::verify;
         interface_list.push_back(this_interface);
         }
      }
   test_invariant();
   return sin;
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;
using libbase::matrix;
using libbase::vector;

#define CONTAINER_TYPE_SEQ \
   (vector)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_concatenated<container,real>
 * where:
 *      container = vector
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_concatenated<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_concatenated<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_concatenated<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            map_concatenated<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
