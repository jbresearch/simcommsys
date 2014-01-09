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

#include "onetimepad.h"
#include <sstream>

namespace libcomm {

using std::cerr;

using libbase::vector;
using libbase::matrix;

// construction and destruction

template <class real>
onetimepad<real>::onetimepad() :
   encoder(NULL)
   {
   }

template <class real>
onetimepad<real>::onetimepad(const fsm& encoder, const int tau,
      const bool terminated, const bool renewable) :
   terminated(terminated), renewable(renewable), encoder(
         dynamic_cast<fsm*> (encoder.clone()))
   {
   const int k = encoder.num_inputs();
   pad.init(tau * k);
   }

template <class real>
onetimepad<real>::onetimepad(const onetimepad& x) :
   terminated(x.terminated), renewable(x.renewable), encoder(
         dynamic_cast<fsm*> (x.encoder->clone())), pad(x.pad), r(x.r)
   {
   }

template <class real>
onetimepad<real>::~onetimepad()
   {
   if (encoder != NULL)
      delete encoder;
   }

// inter-frame operations

template <class real>
void onetimepad<real>::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

template <class real>
void onetimepad<real>::advance()
   {
   static bool initialised = false;

   // do not advance if this interleaver is not renewable
   if (!renewable && initialised)
      return;

   const int m = encoder->mem_order();
   const int k = encoder->num_inputs();
   const int S = encoder->num_symbols();
   const int tau = pad.size() / k;
   // fill in pad
   if (terminated)
      {
      for (int t = 0; t < (tau - m) * k; t++)
         pad(t) = r.ival(S);
      for (int t = (tau - m) * k; t < tau * k; t++)
         pad(t) = fsm::tail;
      // run through the encoder once, so that we work out the tail bits
      encoder->reset();
      for (int t = 0; t < tau; t++)
         {
         vector<int> ip = pad.segment(t * k, k);
         encoder->step(ip);
         }
      }
   else
      {
      for (int t = 0; t < tau * k; t++)
         pad(t) = r.ival(S);
      }

   initialised = true;
   }

// transform functions

template <class real>
void onetimepad<real>::transform(const vector<int>& in, vector<int>& out) const
   {
   const int N = pad.size();
   const int S = encoder->num_symbols();
   assertalways(in.size() == N);
   out.init(in.size());
   for (int i = 0; i < N; i++)
      out(i) = (in(i) + pad(i)) % S;
   }

template <class real>
void onetimepad<real>::transform(const matrix<real>& in, matrix<real>& out) const
   {
   const int N = pad.size();
   const int S = encoder->num_symbols();
   assertalways(in.size().cols() == S);
   assertalways(in.size().rows() == N);
   out.init(in.size());
   for (int i = 0; i < N; i++)
      for (int j = 0; j < S; j++)
         out(i, j) = in(i, (j + pad(i)) % S);
   }

template <class real>
void onetimepad<real>::inverse(const matrix<real>& in, matrix<real>& out) const
   {
   const int N = pad.size();
   const int S = encoder->num_symbols();
   assertalways(in.size().cols() == S);
   assertalways(in.size().rows() == N);
   out.init(in.size());
   for (int i = 0; i < N; i++)
      for (int j = 0; j < S; j++)
         out(i, (j + pad(i)) % S) = in(i, j);
   }

// description output

template <class real>
std::string onetimepad<real>::description() const
   {
   std::ostringstream sout;
   sout << "One-Time-Pad Interleaver (";
   if (terminated)
      sout << "terminated";
   if (terminated && renewable)
      sout << ", ";
   if (renewable)
      sout << "renewable";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& onetimepad<real>::serialize(std::ostream& sout) const
   {
   sout << int(terminated) << std::endl;
   sout << int(renewable) << std::endl;
   sout << pad.size() << std::endl;
   sout << encoder;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& onetimepad<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> terminated >> libbase::verify;
   sin >> libbase::eatcomments >> renewable >> libbase::verify;
   int N;
   sin >> libbase::eatcomments >> N >> libbase::verify;
   pad.init(N);
   sin >> libbase::eatcomments >> encoder >> libbase::verify;
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: onetimepad<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class onetimepad<type>; \
   template <> \
   const serializer onetimepad<type>::shelper( \
         "interleaver", \
         "onetimepad<" BOOST_PP_STRINGIZE(type) ">", \
         onetimepad<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
