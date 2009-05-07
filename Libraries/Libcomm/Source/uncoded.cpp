/*!
   \file

   \section svn Version Control
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
   assertalways(encoder);
   // Check that FSM is memoryless
   assertalways(encoder->mem_order() == 0);
   // since the encoder is memoryless, we can build an input/output table
   lut.init(encoder->num_inputs());
   for(int i=0; i<encoder->num_inputs(); i++)
      lut(i) = encoder->step(i);
   }

void uncoded::free()
   {
   if(encoder != NULL)
      delete encoder;
   }

// constructor / destructor

uncoded::uncoded() :
   encoder(NULL)
   {
   }

uncoded::uncoded(const fsm& encoder, const int tau) :
   tau(tau)
   {
   uncoded::encoder = encoder.clone();
   init();
   }

// internal codec operations

void uncoded::resetpriors()
   {
   }

void uncoded::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Copy the input statistics for the BCJR Algorithm
   for(int t=0; t<This::input_block_size(); t++)
      for(int i=0; i<This::num_inputs(); i++)
         R(t)(i) *= ptable(t)(i);
   }

void uncoded::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Initialize receiver probability vector
   R.init(This::input_block_size());
   for(int t=0; t<This::input_block_size(); t++)
      R(t).init(This::num_inputs());
   // Work out the probabilities of each possible input
   for(int t=0; t<This::input_block_size(); t++)
      for(int x=0; x<This::num_inputs(); x++)
         R(t)(x) = ptable(t)(lut(x));
   }

// encoding and decoding functions

void uncoded::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Initialise result vector
   encoded.init(This::input_block_size());
   // Encode source stream
   for(int t=0; t<This::input_block_size(); t++)
      encoded(t) = lut(source(t));
   }

void uncoded::softdecode(array1vd_t& ri)
   {
   ri = R;
   }

void uncoded::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   assertalways("Not yet implemented");
   }

// description output

std::string uncoded::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded/Repetition Code ("  << This::output_bits() << "," << This::input_bits() << ") - ";
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
