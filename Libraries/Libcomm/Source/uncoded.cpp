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

// initialization / de-allocation

template <class dbl>
void uncoded<dbl>::init()
   {
   assertalways(encoder);
   // Check that FSM is memoryless
   assertalways(encoder->mem_order() == 0);
   // since the encoder is memoryless, we can build an input/output table
   lut.init(encoder->num_inputs());
   for(int i=0; i<encoder->num_inputs(); i++)
      lut(i) = encoder->step(i);
   }

template <class dbl>
void uncoded<dbl>::free()
   {
   if(encoder != NULL)
      delete encoder;
   }

// constructor / destructor

template <class dbl>
uncoded<dbl>::uncoded() :
   encoder(NULL)
   {
   }

template <class dbl>
uncoded<dbl>::uncoded(const fsm& encoder, const int tau) :
   tau(tau)
   {
   uncoded<dbl>::encoder = encoder.clone();
   init();
   }

// internal codec operations

template <class dbl>
void uncoded<dbl>::resetpriors()
   {
   }

template <class dbl>
void uncoded<dbl>::setpriors(const array1vd_t& ptable)
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

template <class dbl>
void uncoded<dbl>::setreceiver(const array1vd_t& ptable)
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

template <class dbl>
void uncoded<dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Initialise result vector
   encoded.init(This::input_block_size());
   // Encode source stream
   for(int t=0; t<This::input_block_size(); t++)
      encoded(t) = lut(source(t));
   }

template <class dbl>
void uncoded<dbl>::softdecode(array1vd_t& ri)
   {
   ri = R;
   }

template <class dbl>
void uncoded<dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   failwith("Not yet implemented");
   }

// description output

template <class dbl>
std::string uncoded<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded/Repetition Code ("  << This::output_bits() << "," << This::input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class dbl>
std::ostream& uncoded<dbl>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   return sout;
   }

// object serialization - loading

template <class dbl>
std::istream& uncoded<dbl>::serialize(std::istream& sin)
   {
   free();
   sin >> encoder;
   sin >> tau;
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class uncoded<float>;
template <>
const serializer uncoded<float>::shelper = serializer("codec", "uncoded<float>", uncoded<float>::create);

template class uncoded<double>;
template <>
const serializer uncoded<double>::shelper = serializer("codec", "uncoded<double>", uncoded<double>::create);

template class uncoded<logrealfast>;
template <>
const serializer uncoded<logrealfast>::shelper = serializer("codec", "uncoded<logrealfast>", uncoded<logrealfast>::create);

}; // end namespace
