/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "uncoded.h"
#include <sstream>

namespace libcomm {

using std::cerr;

const libbase::serializer uncoded::shelper("codec", "uncoded", uncoded::create);

// initialization / de-allocation

void uncoded::init()
   {
   // Check that FSM is memoryless
   assertalways(encoder->mem_order() == 0);
   K = encoder->num_inputs();
   N = encoder->num_outputs();

   // since the encoder is memoryless, we can build an input/output table
   lut.init(K);
   for(int i=0; i<K; i++)
      lut(i) = encoder->step(i);
   }

void uncoded::free()
   {
   if(encoder != NULL)
      delete encoder;
   }

// constructor / destructor

uncoded::uncoded()
   {
   encoder = NULL;
   }

uncoded::uncoded(const fsm& encoder, const int tau)
   {
   uncoded::encoder = encoder.clone();
   uncoded::tau = tau;
   init();
   }

// encoding and decoding functions

void uncoded::encode(array1i_t& source, array1i_t& encoded)
   {
   // Initialise result vector
   encoded.init(tau);
   // Encode source stream
   for(int t=0; t<tau; t++)
      encoded(t) = lut(source(t));
   }

void uncoded::translate(const array2d_t& ptable)
   {
   // Compute factors / sizes & check validity
   const int S = ptable.ysize();
   const int s = int(round(log(double(N))/log(double(S))));
   // Confirm that encoder's output symbols can be represented by
   // an integral number of modulation symbols
   assertalways(N == pow(double(S), s));
   // Confirm input sequence to be of the correct length
   assertalways(ptable.xsize() == tau*s);
   // Initialize results vector
   R.init(tau, K);
   // Work out the probabilities of each possible input
   for(int t=0; t<tau; t++)
      for(int x=0; x<K; x++)
         {
         R(t, x) = 1;
         for(int i=0, thisx = lut(x); i<s; i++, thisx /= S)
            R(t, x) *= ptable(t*s+i, thisx % S);
         }
   }

void uncoded::decode(array2d_t& ri)
   {
   ri = R;
   }

void uncoded::decode(array2d_t& ri, array2d_t& ro)
   {
   assertalways("Not yet implemented");
   }

// description output

std::string uncoded::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded/Repetition Code ("  << output_bits() << "," << input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

std::ostream& uncoded::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   return sout;
   }

// object serialization - loading

std::istream& uncoded::serialize(std::istream& sin)
   {
   free();
   sin >> encoder;
   sin >> tau;
   init();
   return sin;
   }

}; // end namespace
