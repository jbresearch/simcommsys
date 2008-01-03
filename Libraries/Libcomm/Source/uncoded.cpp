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
   if(encoder->mem_order() > 0)
      {
      cerr << "FATAL ERROR (uncoded): cannot use a FSM with memory.\n";
      exit(1);
      }
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

void uncoded::encode(libbase::vector<int>& source, libbase::vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(tau);
   // Encode source stream
   for(int t=0; t<tau; t++)
      encoded(t) = lut(source(t));
   }

void uncoded::translate(const libbase::matrix<double>& ptable)
   {
   // Compute factors / sizes & check validity
   const int S = ptable.ysize();
   const int s = int(round(log(double(N))/log(double(S))));
   if(N != pow(double(S), s))
      {
      cerr << "FATAL ERROR (uncoded): each encoder output (" << N << ") must be";
      cerr << " represented by an integral number of modulation symbols (" << S << ").";
      cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   if(ptable.xsize() != tau*s)
      {
      cerr << "FATAL ERROR (uncoded): demodulation table should have " << tau*s;
      cerr << " symbols, not " << ptable.xsize() << ".\n";
      exit(1);
      }
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

void uncoded::decode(libbase::vector<int>& decoded)
   {
   // Initialise result vector
   decoded.init(tau);
   // Choose the most probable input
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int x=1; x<K; x++)
         if(R(t, x) > R(t, decoded(t)))
            decoded(t) = x;
      }
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
