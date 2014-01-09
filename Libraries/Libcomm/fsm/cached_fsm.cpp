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

#include "cached_fsm.h"

namespace libcomm {

const libbase::serializer cached_fsm::shelper("fsm", "cached_fsm",
      cached_fsm::create);

// Main intialization routine

void cached_fsm::init(fsm& encoder)
   {
   // Initialise internal description of FSM
   k = encoder.num_inputs();
   n = encoder.num_outputs();
   nu = encoder.mem_elements();
   m = encoder.mem_order();
   q = encoder.num_symbols();
   base_description = encoder.description();
   std::ostringstream sout;
   sout << &encoder;
   base_serialization = sout.str();

   // Initialise constants
   const int M = encoder.num_states();
   const int K = encoder.num_input_combinations();
#ifndef NDEBUG
   const int N = encoder.num_output_combinations();
#endif

   // initialise LUT's for state table
   lut_m.init(M, K);
   lut_X.init(M, K);
   lut_Xv.init(M, K);
   for (int mdash = 0; mdash < M; mdash++)
      for (int i = 0; i < K; i++)
         {
         const array1i_t mdash_v = encoder.convert_state(mdash);
         encoder.reset(mdash_v);
         array1i_t input = encoder.convert_input(i);
         const array1i_t output = encoder.step(input);
         lut_Xv(mdash, i) = output;
         assert(lut_Xv(mdash, i).size() == n);
         lut_X(mdash, i) = encoder.convert_output(output);
         assert(lut_X(mdash, i) >= 0 && lut_X(mdash, i) < N);
         lut_m(mdash, i) = encoder.convert_state(encoder.state());
         assert(lut_m(mdash, i) >= 0 && lut_m(mdash, i) < M);
         }

   // initialize state
   reset();
   }

// Serialization

std::ostream& cached_fsm::serialize(std::ostream& sout) const
   {
   sout << "#: Base Encoder" << std::endl;
   sout << base_serialization;
   return sout;
   }

std::istream& cached_fsm::serialize(std::istream& sin)
   {
   fsm *encoder;
   sin >> libbase::eatcomments >> encoder >> libbase::verify;
   init(*encoder);
   delete encoder;
   assertalways(sin.good());
   return sin;
   }

} // end namespace
