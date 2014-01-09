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

#include "sysrepacc.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate encoded output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// encoding and decoding functions

template <class real, class dbl>
void sysrepacc<real, dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   array1i_t parity;
   Base::encode(source, parity);
#if DEBUG>=2
   std::cerr << "Source:" << std::endl;
   source.serialize(std::cerr, '\n');
   std::cerr << "Parity:" << std::endl;
   parity.serialize(std::cerr, '\n');
#endif
   encoded.init(This::output_block_size());
   encoded.segment(0, source.size()).copyfrom(source);
   encoded.segment(source.size(), parity.size()).copyfrom(parity);
   }

template <class real, class dbl>
void sysrepacc<real, dbl>::do_init_decoder(const array1vd_t& ptable)
   {
   // Inherit sizes
   const int Ns = Base::input_block_size();
   const int Np = Base::output_block_size();
   const int q = Base::num_inputs();
   const int qo = Base::num_outputs();
   assertalways(q == qo);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Divide ptable for input and output sides
   const array1vd_t iptable = ptable.extract(0, Ns);
   const array1vd_t optable = ptable.extract(Ns, Np);
   // Perform standard decoder initialization
   Base::init_decoder(optable);
   // Determine intrinsic source statistics (natural)
   // from the channel
   for (int i = 0; i < Ns; i++)
      for (int x = 0; x < q; x++) // 'x' is the input symbol
         rp(i)(x) *= dbl(iptable(i)(x));
   //    BCJR::normalize(rp);
   }

template <class real, class dbl>
void sysrepacc<real, dbl>::do_init_decoder(const array1vd_t& ptable,
      const array1vd_t& app)
   {
   // Inherit sizes
   const int Ns = Base::input_block_size();
   const int Np = Base::output_block_size();
   const int q = Base::num_inputs();
   const int qo = Base::num_outputs();
   assertalways(q == qo);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Divide ptable for input and output sides
   const array1vd_t iptable = ptable.extract(0, Ns);
   const array1vd_t optable = ptable.extract(Ns, Np);
   // Perform standard decoder initialization
   Base::init_decoder(optable, app);
   // Determine intrinsic source statistics (natural)
   // from the channel
   for (int i = 0; i < Ns; i++)
      for (int x = 0; x < q; x++) // 'x' is the input symbol
         rp(i)(x) *= iptable(i)(x);
   //    BCJR::normalize(rp);
   }

// description output

template <class real, class dbl>
std::string sysrepacc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Systematic " << Base::description();
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& sysrepacc<real, dbl>::serialize(std::ostream& sout) const
   {
   return Base::serialize(sout);
   }

// object serialization - loading

template <class real, class dbl>
std::istream& sysrepacc<real, dbl>::serialize(std::istream& sin)
   {
   return Base::serialize(sin);
   }

} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL1_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double) \
   (logrealfast)

/* Serialization string: sysrepacc<real1,real2>
 * where:
 *      real1 = float | double | mpreal | mpgnu | logreal | logrealfast
 *              [real1 is the internal arithmetic type]
 *      real2 = float | double | logrealfast
 *              [real2 is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class sysrepacc<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer sysrepacc<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "sysrepacc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            sysrepacc<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL1_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
