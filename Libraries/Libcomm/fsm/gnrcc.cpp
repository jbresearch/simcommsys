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
 * \see grscc.cpp
 */

#include "gnrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

// FSM helper operations

template <class G>
vector<int> gnrcc<G>::determineinput(const vector<int>& input) const
   {
   vector<int> ip = input;
   for (int i = 0; i < ip.size(); i++)
      if (ip(i) == fsm::tail)
         ip(i) = 0;
   return ip;
   }

template <class G>
vector<G> gnrcc<G>::determinefeedin(const vector<int>& input) const
   {
   for (int i = 0; i < input.size(); i++)
      assert(input(i) != fsm::tail);
   // Convert input to vector of required type
   return vector<G>(input);
   }

// FSM state operations (getting and resetting)

template <class G>
void gnrcc<G>::resetcircular(const vector<int>& zerostate, int n)
   {
   failwith("Function not implemented.");
   }

// Description

template <class G>
std::string gnrcc<G>::description() const
   {
   std::ostringstream sout;
   sout << "NRC code " << ccfsm<G>::description();
   return sout.str();
   }

// Serialization Support

template <class G>
std::ostream& gnrcc<G>::serialize(std::ostream& sout) const
   {
   return ccfsm<G>::serialize(sout);
   }

template <class G>
std::istream& gnrcc<G>::serialize(std::istream& sin)
   {
   return ccfsm<G>::serialize(sin);
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

/* Serialization string: gnrcc<type>
 * where:
 *      type = gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
   template class gnrcc<type>; \
   template <> \
   const serializer gnrcc<type>::shelper( \
         "fsm", \
         "gnrcc<" BOOST_PP_STRINGIZE(type) ">", \
         gnrcc<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, GF_TYPE_SEQ)

} // end namespace
