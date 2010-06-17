/*!
 * \file
 *
 *  Created on: 4 Mar 2010
 *      Author: jabriffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
   }

// Serialization

std::ostream& cached_fsm::serialize(std::ostream& sout) const
   {
   sout << base_serialization;
   return sout;
   }

std::istream& cached_fsm::serialize(std::istream& sin)
   {
   fsm *encoder;
   sin >> libbase::eatcomments >> encoder;
   init(*encoder);
   delete encoder;
   return sin;
   }

} // end namespace
