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

template <class real>
void mapcc<real>::init()
   {
   BCJR::init(*encoder, tau);
   assert(!circular || !endatzero);
   m = endatzero ? encoder->mem_order() : 0;
   M = encoder->num_states();
   K = encoder->num_inputs();
   N = encoder->num_outputs();
   }

template <class real>
void mapcc<real>::free()
   {
   if(encoder != NULL)
      delete encoder;
   }

template <class real>
void mapcc<real>::reset()
   {
   if(circular)
      {
      BCJR::setstart();
      BCJR::setend();
      }
   else if(endatzero)
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

template <class real>
mapcc<real>::mapcc()
   {
   encoder = NULL;
   }

template <class real>
mapcc<real>::mapcc(const fsm& encoder, const int tau, const bool endatzero, const bool circular)
   {
   This::encoder = encoder.clone();
   This::tau = tau;
   This::endatzero = endatzero;
   This::circular = circular;
   init();
   }

// encoding and decoding functions

template <class real>
void mapcc<real>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == input_block_size());
   // Initialise result vector
   encoded.init(tau);
   // Make a local copy of the source, including any necessary tail
   array1i_t source1(tau);
   for(int t=0; t<source.size(); t++)
      source1(t) = source(t);
   for(int t=source.size(); t<tau; t++)
      source1(t) = fsm::tail;
   // Reset the encoder to zero state
   encoder->reset(0);
   // When dealing with a circular system, perform first pass to determine end state,
   // then reset to the corresponding circular state.
   if(circular)
      {
      for(int t=0; t<tau; t++)
         encoder->advance(source1(t));
      encoder->resetcircular();
      }
   // Encode source stream
   for(int t=0; t<tau; t++)
      encoded(t) = encoder->step(source1(t));
   }

template <class real>
void mapcc<real>::translate(const array2d_t& ptable)
   {
   using std::cerr;
   // Compute factors / sizes & check validity
   assert(ptable.size() > 0);
   const int S = ptable(0).size();
   const int s = int(round(log(double(N))/log(double(S))));
   // each encoder output N must be represented by an integral number of
   // modulation symbols
   assertalways(N == pow(double(S), s));
   assertalways(ptable.size() == tau*s);
   // Initialize results vector
   R.init(tau,N);
   // Compute the Input statistics for the BCJR Algorithm
   for(int t=0; t<tau; t++)
      for(int x=0; x<N; x++)
         {
         R(t,x) = 1;
         for(int i=0, thisx = x; i<s; i++, thisx /= S)
            R(t,x) *= ptable(t*s+i)(thisx % S);
         }
   // Reset start- and end-state probabilities
   reset();
   }

template <class real>
void mapcc<real>::softdecode(array2d_t& ri)
   {
   // temporary space to hold complete results (ie. with tail)
   libbase::matrix<double> rif;
   // perform decoding
   BCJR::fdecode(R, rif);
   // remove any tail bits from input set
   ri.init(input_block_size());
   for(int i=0; i<input_block_size(); i++)
      ri(i).init(num_inputs());
   for(int i=0; i<input_block_size(); i++)
      for(int j=0; j<num_inputs(); j++)
         ri(i)(j) = rif(i,j);
   }

template <class real>
void mapcc<real>::softdecode(array2d_t& ri, array2d_t& ro)
   {
   // temporary space to hold complete results (ie. with tail)
   libbase::matrix<double> rif, rof;
   // perform decoding
   BCJR::decode(R, rif, rof);
   // remove any tail bits from input set
   ri.init(input_block_size());
   for(int i=0; i<input_block_size(); i++)
      ri(i).init(num_inputs());
   for(int i=0; i<input_block_size(); i++)
      for(int j=0; j<num_inputs(); j++)
         ri(i)(j) = rif(i,j);
   // copy output set
   ro.init(output_block_size());
   for(int i=0; i<output_block_size(); i++)
      ro(i).init(num_outputs());
   for(int i=0; i<output_block_size(); i++)
      for(int j=0; j<num_outputs(); j++)
         ro(i)(j) = rof(i,j);
   }

// description output

template <class real>
std::string mapcc<real>::description() const
   {
   std::ostringstream sout;
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << "MAP-decoded Convolutional Code (" << output_bits() << "," << input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& mapcc<real>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   sout << int(endatzero) << "\n";
   sout << int(circular) << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& mapcc<real>::serialize(std::istream& sin)
   {
   int temp;
   free();
   sin >> encoder;
   sin >> tau;
   sin >> temp;
   endatzero = temp != 0;
   sin >> temp;
   circular = temp != 0;
   init();
   return sin;
   }

}; // end namespace

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

template class mapcc<double>;
template <>
const serializer mapcc<double>::shelper = serializer("codec", "mapcc<double>", mapcc<double>::create);

template class mapcc<mpreal>;
template <>
const serializer mapcc<mpreal>::shelper = serializer("codec", "mapcc<mpreal>", mapcc<mpreal>::create);

template class mapcc<mpgnu>;
template <>
const serializer mapcc<mpgnu>::shelper = serializer("codec", "mapcc<mpgnu>", mapcc<mpgnu>::create);

template class mapcc<logreal>;
template <>
const serializer mapcc<logreal>::shelper = serializer("codec", "mapcc<logreal>", mapcc<logreal>::create);

template class mapcc<logrealfast>;
template <>
const serializer mapcc<logrealfast>::shelper = serializer("codec", "mapcc<logrealfast>", mapcc<logrealfast>::create);

}; // end namespace
