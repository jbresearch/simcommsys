/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "mapcc.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void mapcc<real, dbl>::init()
   {
   assertalways(encoder);
   BCJR::init(*encoder, tau);
   assertalways(!circular || !endatzero);
   }

template <class real, class dbl>
void mapcc<real, dbl>::free()
   {
   if (encoder != NULL)
      delete encoder;
   }

template <class real, class dbl>
void mapcc<real, dbl>::reset()
   {
   if (circular)
      {
      BCJR::setstart();
      BCJR::setend();
      }
   else if (endatzero)
      {
      BCJR::setstart(0);
      BCJR::setend(0);
      }
   else
      {
      BCJR::setstart(0);
      BCJR::setend();
      }
   }

// constructor / destructor

template <class real, class dbl>
mapcc<real, dbl>::mapcc() :
   encoder(NULL)
   {
   }

template <class real, class dbl>
mapcc<real, dbl>::mapcc(const fsm& encoder, const int tau,
      const bool endatzero, const bool circular) :
   tau(tau), endatzero(endatzero), circular(circular)
   {
   This::encoder = encoder.clone();
   init();
   }

// internal codec functions

template <class real, class dbl>
void mapcc<real, dbl>::resetpriors()
   {
   // Initialize input probability vector
   app.init(This::input_block_size(), This::num_inputs());
   app = 1.0;
   }

template <class real, class dbl>
void mapcc<real, dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Initialize input probability vector
   app.init(This::input_block_size(), This::num_inputs());
   // Copy the input statistics for the BCJR Algorithm
   for (int t = 0; t < app.size().rows(); t++)
      for (int i = 0; i < app.size().cols(); i++)
         app(t, i) = ptable(t)(i);
   }

template <class real, class dbl>
void mapcc<real, dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Initialize receiver probability vector
   R.init(This::output_block_size(), This::num_outputs());
   // Copy the input statistics for the BCJR Algorithm
   for (int t = 0; t < This::output_block_size(); t++)
      for (int x = 0; x < This::num_outputs(); x++)
         R(t, x) = ptable(t)(x);
   // Reset start- and end-state probabilities
   reset();
   }

// encoding and decoding functions

template <class real, class dbl>
void mapcc<real, dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Initialise result vector
   encoded.init(tau);
   // Make a local copy of the source, including any necessary tail
   array1i_t source1(tau);
   for (int t = 0; t < source.size(); t++)
      source1(t) = source(t);
   for (int t = source.size(); t < tau; t++)
      source1(t) = fsm::tail;
   // Reset the encoder to zero state
   encoder->reset(0);
   // When dealing with a circular system, perform first pass to determine end state,
   // then reset to the corresponding circular state.
   if (circular)
      {
      for (int t = 0; t < tau; t++)
         encoder->advance(source1(t));
      encoder->resetcircular();
      }
   // Encode source stream
   for (int t = 0; t < tau; t++)
      encoded(t) = encoder->step(source1(t));
   }

template <class real, class dbl>
void mapcc<real, dbl>::softdecode(array1vd_t& ri)
   {
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif;
   // perform decoding
   BCJR::fdecode(R, app, rif);
   // remove any tail bits from input set
   ri.init(This::input_block_size());
   for (int i = 0; i < This::input_block_size(); i++)
      ri(i).init(This::num_inputs());
   for (int i = 0; i < This::input_block_size(); i++)
      for (int j = 0; j < This::num_inputs(); j++)
         ri(i)(j) = rif(i, j);
   }

template <class real, class dbl>
void mapcc<real, dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif, rof;
   // perform decoding
   BCJR::decode(R, rif, rof);
   // remove any tail bits from input set
   ri.init(This::input_block_size());
   for (int i = 0; i < This::input_block_size(); i++)
      ri(i).init(This::num_inputs());
   for (int i = 0; i < This::input_block_size(); i++)
      for (int j = 0; j < This::num_inputs(); j++)
         ri(i)(j) = rif(i, j);
   // copy output set
   ro.init(This::output_block_size());
   for (int i = 0; i < This::output_block_size(); i++)
      ro(i).init(This::num_outputs());
   for (int i = 0; i < This::output_block_size(); i++)
      for (int j = 0; j < This::num_outputs(); j++)
         ro(i)(j) = rof(i, j);
   }

// description output

template <class real, class dbl>
std::string mapcc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << "MAP-decoded Convolutional Code (" << This::output_bits() << ","
         << This::input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& mapcc<real, dbl>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   sout << int(endatzero) << "\n";
   sout << int(circular) << "\n";
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& mapcc<real, dbl>::serialize(std::istream& sin)
   {
   free();
   sin >> encoder;
   sin >> tau;
   sin >> endatzero;
   sin >> circular;
   init();
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

using libbase::serializer;

template class mapcc<float, float>
template <>
const serializer mapcc<float, float>::shelper = serializer("codec",
      "mapcc<float>", mapcc<float, float>::create);

template class mapcc<double>
template <>
const serializer mapcc<double>::shelper = serializer("codec", "mapcc<double>",
      mapcc<double>::create);

template class mapcc<mpreal>
template <>
const serializer mapcc<mpreal>::shelper = serializer("codec", "mapcc<mpreal>",
      mapcc<mpreal>::create);

template class mapcc<mpgnu>
template <>
const serializer mapcc<mpgnu>::shelper = serializer("codec", "mapcc<mpgnu>",
      mapcc<mpgnu>::create);

template class mapcc<logreal>
template <>
const serializer mapcc<logreal>::shelper = serializer("codec",
      "mapcc<logreal>", mapcc<logreal>::create);

template class mapcc<logrealfast>
template <>
const serializer mapcc<logrealfast>::shelper = serializer("codec",
      "mapcc<logrealfast>", mapcc<logrealfast>::create);

template class mapcc<logrealfast, logrealfast>
template <>
const serializer mapcc<logrealfast, logrealfast>::shelper =
      serializer("codec", "mapcc<logrealfast,logrealfast>", mapcc<logrealfast,
            logrealfast>::create);

} // end namespace
