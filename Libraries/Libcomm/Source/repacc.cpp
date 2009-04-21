/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "repacc.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void repacc<real,dbl>::init()
   {
   assertalways(encoder);
   bcjr<real,dbl>::init(*encoder, output_block_size().x);
   // check that encoder is rate-1
   assertalways(num_inputs() == num_outputs());
   assertalways(inter);
   // TODO: check interleaver sizes
   assertalways(iter > 0);

   initialised = false;
   }

template <class real, class dbl>
void repacc<real,dbl>::free()
   {
   if(encoder != NULL)
      delete encoder;
   if(inter != NULL)
      delete inter;
   }

template <class real, class dbl>
void repacc<real,dbl>::reset()
   {
   if(endatzero)
      {
      bcjr<real,dbl>::setstart(0);
      bcjr<real,dbl>::setend(0);
      }
   else
      {
      bcjr<real,dbl>::setstart(0);
      bcjr<real,dbl>::setend();
      }
   }

// constructor / destructor

template <class real, class dbl>
repacc<real,dbl>::repacc()
   {
   encoder = NULL;
   inter = NULL;
   }

// encoding and decoding functions

template <class real, class dbl>
void repacc<real,dbl>::seedfrom(libbase::random& r)
   {
   inter->seedfrom(r);
   }

template <class real, class dbl>
void repacc<real,dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == input_block_size());
   // Compute repeater output, including any necessary tail
   array1i_t rep(output_block_size());
   for(int i=0; i<source.size(); i++)
      for(int j=0; j<q; j++)
         rep(i*q+j) = source(i);
   for(int i=source.size()*q; i<output_block_size(); i++)
      rep(i) = fsm::tail;

   // Declare space for the interleaved sequence
   array1i_t rep2;
   // Advance interleaver to the next block
   inter->advance();
   // Create interleaved sequence
   inter->transform(rep, rep2);

   // Initialise result vector
   encoded.init(output_block_size());
   // Reset the encoder to zero state
   encoder->reset(0);
   // Encode sequence
   for(int i=0; i<output_block_size(); i++)
      encoded(i) = encoder->step(rep2(i)) / num_inputs();
   // check that encoder finishes correctly
   if(endatzero)
      assertalways(encoder->state() == 0);
   }

/*! \copydoc codec::translate()

   Sets: ra, R

   \note The BCJR normalization method is used to normalize the channel-derived
         (intrinsic) probabilities 'r' and 'R'; in view of this, the a-priori
         probabilities are now created normalized.
*/
template <class real, class dbl>
void repacc<real,dbl>::translate(const libbase::vector< libbase::vector<double> >& ptable)
   {
   assertalways(ptable.size() > 0);
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable(0).size() == num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == output_block_size());

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // Get the necessary data from the channel
   for(int i=0; i<output_block_size(); i++)
      for(int x=0; x<num_inputs(); x++)
         R(i, x) = ptable(i)(x);
   bcjr<real,dbl>::normalize(R);

   // Initialise a priori probabilities (extrinsic)
   ra = 1.0;

   // Reset start- and end-state probabilities
   reset();
   }

template <class real, class dbl>
void repacc<real,dbl>::softdecode(array1vd_t& ri)
   {
   /*
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif;
   // after working all sets, ri is the intrinsic+extrinsic information
   // from the last stage decoder.
   for(int set=0; set<num_sets(); set++)
      {
      bcjr_wrap(set, ra, rif, ra);
      bcjr<real,dbl>::normalize(ra);
      }
   bcjr<real,dbl>::normalize(rif);
   // remove any tail bits from input set
   ri.init(input_block_size());
   for(int i=0; i<input_block_size(); i++)
      ri(i).init(num_inputs());
   for(int i=0; i<input_block_size(); i++)
      for(int j=0; j<num_inputs(); j++)
         ri(i)(j) = rif(i,j);
   */
   }

template <class real, class dbl>
void repacc<real,dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   assertalways("Not yet implemented");
   }

// description output

template <class real, class dbl>
std::string repacc<real,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Repeat-Accumulate Code (" << N << "," << q << ") - ";
   sout << encoder->description() << ", ";
   sout << inter->description() << ", ";
   sout << iter << " iterations, ";
   sout << (endatzero ? "Terminated" : "Unterminated");
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& repacc<real,dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << 1 << '\n';
   sout << encoder;
   sout << inter;
   sout << N << '\n';
   sout << q << '\n';
   sout << iter << '\n';
   sout << int(endatzero) << '\n';
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& repacc<real,dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> version;
   // get first-version items
   sin >> encoder;
   sin >> inter;
   sin >> N;
   sin >> q;
   sin >> iter;
   sin >> endatzero;
   init();
   assertalways(sin.good());
   return sin;
   }

}; // end namespace
