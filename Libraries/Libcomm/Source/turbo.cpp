/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "turbo.h"
#include "flat.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void turbo<real,dbl>::init()
   {
   bcjr<real,dbl>::init(*encoder, tau);

   assertalways(enc_parity()*num_inputs() == enc_outputs());
   assertalways(num_sets() > 0);
   assertalways(tau > 0);
   // TODO: check interleavers
   assertalways(!endatzero || !circular);
   assertalways(iter > 0);

   initialised = false;
   }

template <class real, class dbl>
void turbo<real,dbl>::free()
   {
   if(encoder != NULL)
      delete encoder;
   for(int i=0; i<inter.size(); i++)
      delete inter(i);
   }

template <class real, class dbl>
void turbo<real,dbl>::reset()
   {
   if(circular)
      {
      assert(initialised);
      for(int set=0; set<num_sets(); set++)
         {
         ss(set) = dbl(1.0/double(enc_states()));
         se(set) = dbl(1.0/double(enc_states()));
         }
      }
   else if(endatzero)
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
turbo<real,dbl>::turbo()
   {
   encoder = NULL;
   }

template <class real, class dbl>
turbo<real,dbl>::turbo(const fsm& encoder, const int tau, \
   const libbase::vector<interleaver<dbl> *>& inter, const int iter, \
   const bool endatzero, const bool parallel, const bool circular)
   {
   turbo::encoder = encoder.clone();
   turbo::tau = tau;
   turbo::inter = inter;
   turbo::endatzero = endatzero;
   turbo::parallel = parallel;
   turbo::circular = circular;
   turbo::iter = iter;
   init();
   }

// memory allocator (for internal use only)

template <class real, class dbl>
void turbo<real,dbl>::allocate()
   {
   R.init(num_sets());
   for(int i=0; i<num_sets(); i++)
      R(i).init(tau, enc_outputs());

   rp.init(tau, num_inputs());
   ri.init(tau, num_inputs());
   rai.init(tau, num_inputs());
   rii.init(tau, num_inputs());

   if(parallel)
      {
      ra.init(num_sets());
      for(int i=0; i<num_sets(); i++)
         ra(i).init(tau, num_inputs());
      }
   else
      {
      ra.init(1);
      ra(0).init(tau, num_inputs());
      }

   if(circular)
      {
      ss.init(num_sets());
      se.init(num_sets());
      for(int i=0; i<num_sets(); i++)
         {
         ss(i).init(enc_states());
         se(i).init(enc_states());
         }
      }

   // determine memory occupied and tell user
   std::ios::fmtflags flags = std::cerr.flags();
   std::cerr << "Turbo Memory Usage: " << std::fixed << std::setprecision(1);
   std::cerr << sizeof(dbl)*( rp.size() + ri.size() + rai.size() + rii.size()
                           + R.size()*R(0).size() + ra.size()*ra(0).size()
                           //+ ss.size()*ss(0).size() + se.size()*se(0).size()
                           )/double(1<<20) << "MB\n";
   std::cerr.setf(flags);
   // flag the state of the arrays
   initialised = true;
   }

// wrapping functions

/*!
   \brief Computes extrinsic probabilities
   \param[in]  ra  A-priori (extrinsic) probabilities of input values
   \param[in]  ri  A-posteriori probabilities of input values
   \param[in]  r   A-priori intrinsic probabilities of input values
   \param[out] re  Extrinsic probabilities of input values

   \note It is counter-productive to vectorize this, as it would require
         many unnecessary temporary matrix creations.

   \note Before the code review of v2.72, the division was only computed
         at matrix elements where the corresponding 'ri' was greater than
         zero. We have no idea why this was done - will need to check old
         documentation. There seems to be marginal effect on results/speed,
         so the natural (no-check) computation was restored. Old code was:
         \code
         if(ri(t, x) > dbl(0))
            re(t, x) = ri(t, x) / (ra(t, x) * r(t, x));
         else
            re(t, x) = 0;
         \endcode

   \warning The return matrix re may actually be one of the input matrices,
            so one must be careful not to overwrite positions that still
            need to be read.
*/
template <class real, class dbl>
void turbo<real,dbl>::work_extrinsic(const array2d_t& ra, const array2d_t& ri, const array2d_t& r, array2d_t& re)
   {
   // Determine sizes from input matrix
   const int tau = ri.xsize();
   const int K = ri.ysize();
   // Check all matrices are the right size
   assert(ra.xsize() == tau && ra.ysize() == K);
   assert(r.xsize() == tau && r.ysize() == K);
   assert(re.xsize() == tau && re.ysize() == K);
   // Compute extrinsic values
   for(int t=0; t<tau; t++)
      for(int x=0; x<K; x++)
         re(t, x) = ri(t, x) / (ra(t, x) * r(t, x));
   }

/*!
   \brief Preparation for BCJR decoding
   \param[in]  set Parity sequence being decoded
   \param[in]  ra  A-priori (extrinsic) probabilities of input values
   \param[out] rai Interleaved version of a-priori probabilities 

   This method does the preparatory work required before BCJR decoding,
   including start/end state probability setting for circular decoding, and
   pre-interleaving of a-priori probabilities.

   \note When using a circular trellis, the start- and end-state probabilities
         are re-initialize with the stored values from the previous turn.
*/
template <class real, class dbl>
void turbo<real,dbl>::bcjr_pre(const int set, const array2d_t& ra, array2d_t& rai)
   {
   if(circular)
      {
      bcjr<real,dbl>::setstart(ss(set));
      bcjr<real,dbl>::setend(se(set));
      }
   inter(set)->transform(ra, rai);
   }

/*!
   \brief Post-processing for BCJR decoding
   \param[in]  set Parity sequence being decoded
   \param[in]  rii Interleaved version of a-posteriori probabilities 
   \param[out] ri  A-posteriori probabilities of input values

   This method does the post-processing work required after BCJR decoding,
   including start/end state probability storing for circular decoding, and
   de-interleaving of a-posteriori probabilities.

   \note When using a circular trellis, the start- and end-state probabilities
         are stored for the next turn.
*/
template <class real, class dbl>
void turbo<real,dbl>::bcjr_post(const int set, const array2d_t& rii, array2d_t& ri)
   {
   inter(set)->inverse(rii, ri);
   if(circular)
      {
      ss(set) = bcjr<real,dbl>::getstart();
      se(set) = bcjr<real,dbl>::getend();
      }
   }

/*!
   \brief Complete BCJR decoding cycle
   \param[in]  set Parity sequence being decoded
   \param[in]  ra  A-priori (extrinsic) probabilities of input values
   \param[out] ri  A-posteriori probabilities of input values
   \param[out] re  Extrinsic probabilities of input values (will be used later
                   as the new 'a-priori' probabilities)

   This method performs a complete decoding cycle, including start/end state
   probability settings for circular decoding, and any interleaving/de-interleaving.

   \warning The return matrix re may actually be the input matrix ra,
            so one must be careful not to overwrite positions that still
            need to be read.
*/
template <class real, class dbl>
void turbo<real,dbl>::bcjr_wrap(const int set, const array2d_t& ra, array2d_t& ri, array2d_t& re)
   {
   bcjr_pre(set, ra, rai);
   bcjr<real,dbl>::fdecode(R(set), rai, rii);
   bcjr_post(set, rii, ri);
   work_extrinsic(ra, ri, rp, re);
   }

/*! \brief Perform a complete serial-decoding cycle

   \note The BCJR normalization method is used to normalize the extrinsic
         probabilities.
*/
template <class real, class dbl>
void turbo<real,dbl>::decode_serial(array2d_t& ri)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();
   // initialise results matrix
   ri.init(tau, num_inputs());
   // after working all sets, ri is the intrinsic+extrinsic information
   // from the last stage decoder.
   for(int set=0; set<num_sets(); set++)
      {
      bcjr_wrap(set, ra(0), ri, ra(0));
      bcjr<real,dbl>::normalize(ra(0));
      }
   bcjr<real,dbl>::normalize(ri);
   }

/*! \brief Perform a complete parallel-decoding cycle

   \note The BCJR normalization method is used to normalize the extrinsic
         probabilities.

   \warning Simulations show that parallel-decoding works well with the
            1/4-rate, 3-code, K=3 (111/101), N=4096 code from divs95b;
            however, when simulating larger codes (N=8192) the system seems
            to go unstable after a few iterations. Also significantly, similar
            codes with lower rates (1/6 and 1/8) perform _worse_ as the rate
            decreases.
*/
template <class real, class dbl>
void turbo<real,dbl>::decode_parallel(array2d_t& ri)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();
   // initialise results matrix
   ri.init(tau, num_inputs());
   // here ri is only a temporary space
   // and ra(set) is updated with the extrinsic information for that set
   for(int set=0; set<num_sets(); set++)
      bcjr_wrap(set, ra(set), ri, ra(set));
   // the following are repeated at each frame element, for each possible symbol
   // work in ri the sum of all extrinsic information
   ri = ra(0);
   for(int set=1; set<num_sets(); set++)
      ri.multiplyby(ra(set));
   // compute the next-stage a priori information by subtracting the extrinsic
   // information of the current stage from the sum of all extrinsic information.
   for(int set=0; set<num_sets(); set++)
      ra(set) = ri.divide(ra(set));
   // add the channel information to the sum of extrinsic information
   ri.multiplyby(rp);
   // normalize results
   for(int set=0; set<num_sets(); set++)
      bcjr<real,dbl>::normalize(ra(set));
   bcjr<real,dbl>::normalize(ri);
   }

// encoding and decoding functions

template <class real, class dbl>
void turbo<real,dbl>::seedfrom(libbase::random& r)
   {
   for(int set=0; set<num_sets(); set++)
      inter(set)->seedfrom(r);
   }

template <class real, class dbl>
void turbo<real,dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == input_block_size());
   // Initialise result vector
   encoded.init(tau);
   // Allocate space for the encoder outputs
   libbase::matrix<int> x(num_sets(), tau);
   // Make a local copy of the source, including any necessary tail
   array1i_t source1(tau);
   for(int t=0; t<source.size(); t++)
      source1(t) = source(t);
   for(int t=source.size(); t<tau; t++)
      source1(t) = fsm::tail;
   // Declare space for the interleaved source
   array1i_t source2;

   // Consider sets in order
   for(int set=0; set<num_sets(); set++)
      {
      // Advance interleaver to the next block
      inter(set)->advance();
      // Create interleaved version of source
      inter(set)->transform(source1, source2);

      // Reset the encoder to zero state
      encoder->reset(0);

      // When dealing with a circular system, perform first pass to determine end state,
      // then reset to the corresponding circular state.
      int cstate = 0;
      if(circular)
         {
         for(int t=0; t<tau; t++)
            encoder->advance(source2(t));
         encoder->resetcircular();
         cstate = encoder->state();
         }

      // Encode source (non-interleaved must be done first to determine tail bit values)
      for(int t=0; t<tau; t++)
         x(set, t) = encoder->step(source2(t)) / num_inputs();

      // If this was the first (non-interleaved) set, copy back the source
      // to fix the tail bit values, if any
      if(endatzero && set == 0)
         source1 = source2;

      // check that encoder finishes correctly
      if(circular)
         assertalways(encoder->state() == cstate);
      if(endatzero)
         assertalways(encoder->state() == 0);
      }

   // Encode source stream
   for(int t=0; t<tau; t++)
      {
      // data bits
      encoded(t) = source1(t);
      // parity bits
      for(int set=0, mul=num_inputs(); set<num_sets(); set++, mul*=enc_parity())
         encoded(t) += x(set, t)*mul;
      }
   }

/*! \copydoc codec::translate()

   \note The BCJR normalization method is used to normalize the channel-derived
         (intrinsic) probabilities 'r' and 'R'; in view of this, the a-priori
         probabilities are now created normalized.

   \todo Move temporary matrix to a class member (consider if this will
         actually constitute a speedup)
*/
template <class real, class dbl>
void turbo<real,dbl>::translate(const libbase::matrix<double>& ptable)
   {
   // Compute factors / sizes & check validity
   const int S = ptable.ysize();
   const int sp = int(round(log(double(enc_parity()))/log(double(S))));
   const int sk = int(round(log(double(num_inputs()))/log(double(S))));
   const int s = sk + num_sets()*sp;
   // Confirm that encoder's parity and input symbols can be represented by
   // an integral number of modulation symbols
   assertalways(enc_parity() == pow(double(S), sp));
   assertalways(num_inputs() == pow(double(S), sk));
   // Confirm input sequence to be of the correct length
   assertalways(ptable.xsize() == tau*s);

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // Allocate space for temporary matrices
   libbase::matrix3<dbl> p(num_sets(), tau, enc_parity());

   // Get the necessary data from the channel
   for(int t=0; t<tau; t++)
      {
      // Input (data) bits [set 0 only]
      for(int x=0; x<num_inputs(); x++)
         {
         rp(t, x) = 1;
         for(int i=0, thisx = x; i<sk; i++, thisx /= S)
            rp(t, x) *= ptable(t*s+i, thisx % S);
         }
      // Parity bits [all sets]
      for(int x=0; x<enc_parity(); x++)
         for(int set=0, offset=sk; set<num_sets(); set++)
            {
            p(set, t, x) = 1;
            for(int i=0, thisx = x; i<sp; i++, thisx /= S)
               p(set, t, x) *= ptable(t*s+i+offset, thisx % S);
            offset += sp;
            }
      }

   // Initialise a priori probabilities (extrinsic)
   for(int set=0; set<(parallel ? num_sets() : 1); set++)
      ra(set) = 1.0;

   // Normalize a priori probabilities (intrinsic - source)
   bcjr<real,dbl>::normalize(rp);

   // Compute and normalize a priori probabilities (intrinsic - encoded)
   array2d_t rpi;
   for(int set=0; set<num_sets(); set++)
      {
      inter(set)->transform(rp, rpi);
      for(int t=0; t<tau; t++)
         for(int x=0; x<enc_outputs(); x++)
            R(set)(t, x) = rpi(t, x%num_inputs()) * p(set, t, x/num_inputs());
      bcjr<real,dbl>::normalize(R(set));
      }

   // Reset start- and end-state probabilities
   reset();
   }

template <class real, class dbl>
void turbo<real,dbl>::decode(array2d_t& ri)
   {
   // do one iteration, in serial or parallel as required
   if(parallel)
      decode_parallel(ri);
   else
      decode_serial(ri);
   }

template <class real, class dbl>
void turbo<real,dbl>::decode(array2d_t& ri, array2d_t& ro)
   {
   }

// description output

template <class real, class dbl>
std::string turbo<real,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Turbo Code (" << this->output_bits() << "," << this->input_bits() << ") - ";
   sout << encoder->description() << ", ";
   for(int i=0; i<inter.size(); i++)
      sout << inter(i)->description() << ", ";
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << (parallel ? "Parallel Decoding, " : "Serial Decoding, ");
   sout << iter << " iterations";
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& turbo<real,dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << 1 << '\n';
   sout << encoder;
   sout << tau << '\n';
   sout << num_sets() << '\n';
   for(int i=0; i<inter.size(); i++)
      sout << inter(i);
   sout << int(endatzero) << '\n';
   sout << int(circular) << '\n';
   sout << int(parallel) << '\n';
   sout << iter << '\n';
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& turbo<real,dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> version;
   // handle old-format files
   if(sin.fail())
      {
      version = 0;
      sin.clear();
      }
   sin >> encoder;
   sin >> tau;
   int sets;
   sin >> sets;
   inter.init(sets);
   if(version == 0)
      {
      inter(0) = new flat<dbl>(tau);
      for(int i=1; i<inter.size(); i++)
         sin >> inter(i);
      }
   else
      {
      for(int i=0; i<inter.size(); i++)
         sin >> inter(i);
      }
   sin >> endatzero;
   sin >> circular;
   sin >> parallel;
   sin >> iter;
   init();
   assertalways(sin.good());
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

template class turbo<double>;
template <>
const serializer turbo<double>::shelper = serializer("codec", "turbo<double>", turbo<double>::create);

template class turbo<mpreal>;
template <>
const serializer turbo<mpreal>::shelper = serializer("codec", "turbo<mpreal>", turbo<mpreal>::create);

template class turbo<mpgnu>;
template <>
const serializer turbo<mpgnu>::shelper = serializer("codec", "turbo<mpgnu>", turbo<mpgnu>::create);

template class turbo<logreal>;
template <>
const serializer turbo<logreal>::shelper = serializer("codec", "turbo<logreal>", turbo<logreal>::create);

template class turbo<logrealfast>;
template <>
const serializer turbo<logrealfast>::shelper = serializer("codec", "turbo<logrealfast>", turbo<logrealfast>::create);

template class turbo<logrealfast,logrealfast>;
template <>
const serializer turbo<logrealfast,logrealfast>::shelper = serializer("codec", "turbo<logrealfast,logrealfast>", turbo<logrealfast,logrealfast>::create);

}; // end namespace
