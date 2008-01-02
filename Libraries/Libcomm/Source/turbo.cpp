/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "turbo.h"
#include <sstream>

#ifdef _DEBUG
#  define DEBUG
#endif

namespace libcomm {

using std::cerr;
using libbase::trace;
using libbase::vector;
using libbase::matrix;
using libbase::matrix3;

// initialization / de-allocation

template <class real, class dbl> void turbo<real,dbl>::init()
   {
   bcjr<real,dbl>::init(*encoder, tau, !circular, endatzero, circular);

   m = endatzero ? encoder->mem_order() : 0;
   M = encoder->num_states();
   K = encoder->num_inputs();
   N = encoder->num_outputs();
   P = N/K;             // this *must* be an integer (for any binary code, at least)

   seed(0);
   initialised = false;
   }

template <class real, class dbl> void turbo<real,dbl>::free()
   {
   if(encoder != NULL)
      delete encoder;
   for(int i=0; i<inter.size(); i++)
      delete inter(i);
   }

// constructor / destructor

template <class real, class dbl> turbo<real,dbl>::turbo()
   {
   encoder = NULL;
   }

template <class real, class dbl> turbo<real,dbl>::turbo(const fsm& encoder, const int tau, \
   const vector<interleaver *>& inter, const int iter, const bool simile, \
   const bool endatzero, const bool parallel, const bool circular)
   {
   turbo::encoder = encoder.clone();
   turbo::tau = tau;
   turbo::sets = inter.size()+1;
   turbo::inter = inter;
   turbo::simile = simile;
   turbo::endatzero = endatzero;
   turbo::parallel = parallel;
   turbo::circular = circular;
   turbo::iter = iter;
   init();
   }

// memory allocator (for internal use only)

template <class real, class dbl> void turbo<real,dbl>::allocate()
   {
   r.init(sets);
   R.init(sets);
   for(int i=0; i<sets; i++)
      {
      r(i).init(tau, K);
      R(i).init(tau, N);
      }

   ri.init(tau, K);
   rai.init(tau, K);
   rii.init(tau, K);

   if(parallel)
      {
      ra.init(sets);
      for(int i=0; i<sets; i++)
         ra(i).init(tau, K);
      }
   else
      {
      ra.init(1);
      ra(0).init(tau, K);
      }

   initialised = true;
   }

// wrapping functions

template <class real, class dbl> void turbo<real,dbl>::work_extrinsic(const matrix<dbl>& ra, const matrix<dbl>& ri, const matrix<dbl>& r, matrix<dbl>& re)
   {
   // calculate extrinsic information
   for(int t=0; t<tau; t++)
      for(int x=0; x<K; x++)
         if(ri(t, x) > dbl(0))
            re(t, x) = ri(t, x) / (ra(t, x) * r(t, x));
         else
            re(t, x) = 0;
   }

// Static:
// R(set) = table of probabilities of each possible encoder output at each timestep
// Inputs:
// ra = table of a-priori probabilities of input values at each timestep
// Outputs:
// ri = table of a-posteriori probabilities of input values at each timestep
// re = table of extrinsic probabilities of input values at each timestep
//      (these will be later used as the new 'a-priori' probabilities)
template <class real, class dbl> void turbo<real,dbl>::bcjr_wrap(const int set, const matrix<dbl>& ra, matrix<dbl>& ri, matrix<dbl>& re)
   {
   trace << "DEBUG (turbo): bcjr_wrap - set=" << set << ", ra=" << &ra << ", ri=" << &ri << ", re=" << &re;
   trace << ", ra(mean) = " << ra.mean();
   // pass through BCJR algorithm
   // interleaving and de-interleaving is performed except for the first set
   if(set == 0)
      {
      bcjr<real,dbl>::fdecode(R(set), ra, ri);
      work_extrinsic(ra, ri, r(set), re);
      }
   else
      {
      inter(set-1)->transform(ra, rai);
      bcjr<real,dbl>::fdecode(R(set), rai, rii);
      work_extrinsic(rai, rii, r(set), rai);
      inter(set-1)->inverse(rii, ri);
      inter(set-1)->inverse(rai, re);
      }
   trace << ", ri(mean) = " << ri.mean() << ", re(mean) = " << re.mean() << ".\n";
   }

template <class real, class dbl> void turbo<real,dbl>::hard_decision(const matrix<dbl>& ri, vector<int>& decoded)
   {
   // Decide which input sequence was most probable.
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(ri(t, i) > ri(t, decoded(t)))
            decoded(t) = i;
      }
#ifdef DEBUG
   static int iter=0;
   const int ones = decoded.sum();
   trace << "DEBUG (turbo): iter=" << iter \
      << ", decoded ones = " << ones << "/" << tau \
      << ", ri(mean) = " << ri.mean() \
      << ", r(0)(mean) = " << r(0).mean() << "\n";
   if(fabs(ones/double(tau) - 0.5) > 0.05)
      {
      trace << "DEBUG (turbo): decoded = " << decoded << "\n";
      trace << "DEBUG (turbo): ri = " << ri << "\n";
      }
   iter++;
#endif
   }

template <class real, class dbl> void turbo<real,dbl>::decode_serial(matrix<dbl>& ri)
   {
   // after working all sets, ri is the intrinsic+extrinsic information
   // from the last stage decoder.
   for(int set=0; set<sets; set++)
      {
      bcjr_wrap(set, ra(0), ri, ra(0));
      bcjr<real,dbl>::normalize(ra(0));
      }
   bcjr<real,dbl>::normalize(ri);
   }

template <class real, class dbl> void turbo<real,dbl>::decode_parallel(matrix<dbl>& ri)
   {
   // here ri is only a temporary space
   // and ra(set) is updated with the extrinsic information for that set
   for(int set=0; set<sets; set++)
      bcjr_wrap(set, ra(set), ri, ra(set));
   // repeat, at each frame element, for each possible symbol
   for(int t=0; t<tau; t++)
      for(int x=0; x<K; x++)
         {
         // work in ri the sum of all extrinsic information
         {
         ri(t, x) = 1;
         for(int set=0; set<sets; set++)
            ri(t, x) *= ra(set)(t, x);
         }
         // compute the next-stage a priori information by subtracting the extrinsic
         // information of the current stage from the sum of all extrinsic information.
         {
         for(int set=0; set<sets; set++)
            ra(set)(t, x) = ri(t, x) / ra(set)(t, x);
         }
         // add the channel information to the sum of extrinsic information
         ri(t, x) *= r(0)(t, x);
         }
   // normalize results
   {
   for(int set=0; set<sets; set++)
      bcjr<real,dbl>::normalize(ra(set));
   }
   bcjr<real,dbl>::normalize(ri);
   }

// encoding and decoding functions

template <class real, class dbl> void turbo<real,dbl>::seed(const int s)
   {
   for(int set=1; set<sets; set++)
      inter(set-1)->seed(s+set);
   }

template <class real, class dbl> void turbo<real,dbl>::encode(vector<int>& source, vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(tau);

   // Allocate space for the encoder outputs
   matrix<int> x(sets, tau);
   // Allocate space for the interleaved sources
   vector<int> source2(tau);

   // Consider sets in order
   {
   for(int set=0; set<sets; set++)
      {
      // For first set, copy original source
      if(set == 0)
         source2 = source;
      else
         {
         // Advance interleaver to the next block
         inter(set-1)->advance();
         // Create interleaved version of source
         inter(set-1)->transform(source, source2);
         }

      // Reset the encoder to zero state
      encoder->reset(0);

      // When dealing with a circular system, perform first pass to determine end state,
      // then reset to the corresponding circular state.
      int cstate;
      if(circular)
         {
         for(int t=0; t<tau; t++)
            encoder->advance(source2(t));
         encoder->resetcircular();
         cstate = encoder->state();
         }

      // Encode source (non-interleaved must be done first to determine tail bit values)
      for(int t=0; t<tau; t++)
         x(set, t) = encoder->step(source2(t)) / K;

      // If this was the first (non-interleaved) set, copy back the source
      // to fix the tail bit values, if any
      if(set == 0)
         source = source2;

      // check that encoder finishes in circulation state (applies for all interleavers)
      if(circular && encoder->state() != cstate)
         {
         cerr << "FATAL ERROR (turbo): Invalid finishing state for set " << set << " encoder - " << encoder->state() << " (should be " << cstate << ")\n";
         exit(1);
         }

      // check that encoder finishes in state zero (applies for all interleavers)
      if(endatzero && encoder->state() != 0)
         {
         cerr << "FATAL ERROR (turbo): Invalid finishing state for set " << set << " encoder - " << encoder->state() << "\n";
         exit(1);
         }
      }
   }

   // Encode source stream
   for(int t=0; t<tau; t++)
      {
      // data bits
      encoded(t) = source(t);
      // parity bits
      for(int set=0, mul=K; set<sets; set++, mul*=P)
         encoded(t) += x(set, t)*mul;
      }
   }

template <class real, class dbl> void turbo<real,dbl>::translate(const matrix<double>& ptable)
   {
   // Compute factors / sizes & check validity
   const int S = ptable.ysize();
   const int sp = int(libbase::round(log(double(P))/log(double(S))));
   const int sk = int(libbase::round(log(double(K))/log(double(S))));
   const int s = sk + sets*sp;
   if(P != pow(double(S), sp) || K != pow(double(S), sk))
      {
      cerr << "FATAL ERROR (turbo): each encoder parity (" << P << ") and input (" << K << ")";
      cerr << " must be represented by an integral number of modulation symbols (" << S << ").";
      cerr << " Suggested number of mod. symbols/encoder input and parity were (" << sp << "," << sk << ").\n";
      exit(1);
      }
   if(ptable.xsize() != tau*s)
      {
      cerr << "FATAL ERROR (turbo): demodulation table should have " << tau*s;
      cerr << " symbols, not " << ptable.xsize() << ".\n";
      exit(1);
      }

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // Allocate space for temporary matrices
   matrix3<dbl> p(sets, tau, P);

   // Get the necessary data from the channel
   for(int t=0; t<tau; t++)
      {
      // Input (data) bits [set 0 only]
      for(int x=0; x<K; x++)
         {
         r(0)(t, x) = 1;
         for(int i=0, thisx = x; i<sk; i++, thisx /= S)
            r(0)(t, x) *= ptable(t*s+i, thisx % S);
         }
      // Parity bits [all sets]
      {
      for(int x=0; x<P; x++)
         for(int set=0, offset=sk; set<sets; set++)
            {
            p(set, t, x) = 1;
            for(int i=0, thisx = x; i<sp; i++, thisx /= S)
               p(set, t, x) *= ptable(t*s+i+offset, thisx % S);
            offset += sp;
            }
      }
      }

   // Handle tail parity for simile interleavers
   static bool shown = false;
   if(simile && !shown)
      {
      cerr << "WARNING: code for handling simile interleavers is commented out.\n";
      shown = true;
      }
      /*
      {
      // check if the puncturing pattern is odd/even in the tail section
      bool is_stippled = true;
      for(int t=tau-m; t<tau; t++)
         for(int set=1; set<sets; set++)
            if(punc->transmit(set,t) != ((set-1)%(s-1) == t%(s-1)))
               is_stippled = false;

      // copy over the probabilities for the punctured bits from unpunctured ones
      if(is_stippled)
         {
         static bool print_debug = false;
         if(!print_debug)
            {
            cerr << "DEBUG: doing modifier for simile interleavers with stippled puncturing.\n";
            print_debug = true;
            }
         for(int t=tau-m; t<tau; t++)
            {
            int base = t%sets;
            for(int set=1; set<sets; set++)
               for(int x=0; x<P; x++)
                  p((base+set)%sets, t, x) = p(base, t, x);
            }
         }
      }
      */

   // Initialise a priori probabilities (extrinsic)
   for(int set=0; set<(parallel ? sets : 1); set++)
      for(int t=0; t<tau; t++)
         for(int x=0; x<K; x++)
            ra(set)(t, x) = 1.0;

   // Normalize and compute a priori probabilities (intrinsic - source)
   {
   bcjr<real,dbl>::normalize(r(0));
   for(int set=1; set<sets; set++)
      inter(set-1)->transform(r(0), r(set));
   }

   // Compute and normalize a priori probabilities (intrinsic - encoded)
   {
   for(int set=0; set<sets; set++)
      {
      for(int t=0; t<tau; t++)
         for(int x=0; x<N; x++)
            R(set)(t, x) = r(set)(t, x%K) * p(set, t, x/K);
      bcjr<real,dbl>::normalize(R(set));
      }
   }

   // Reset start- and end-state probabilities
   bcjr<real,dbl>::reset();
   }

template <class real, class dbl> void turbo<real,dbl>::decode(vector<int>& decoded)
   {
   // Initialise result vector
   decoded.init(tau);

   // initialise memory if necessary
   if(!initialised)
      allocate();

   // do one iteration, in serial or parallel as required
   if(parallel)
      decode_parallel(ri);
   else
      decode_serial(ri);

   // Decide which input sequence was most probable, based on BCJR stats.
   hard_decision(ri, decoded);
   }

// description output

template <class real, class dbl> std::string turbo<real,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Turbo Code (" << output_bits() << "," << input_bits() << ") - ";
   sout << encoder->description() << ", ";
   for(int i=0; i<inter.size(); i++)
      sout << inter(i)->description() << ", ";
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (simile ? "Simile, " : "Non-simile, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << (parallel ? "Parallel Decoding, " : "Serial Decoding, ");
   sout << iter << " iterations";
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl> std::ostream& turbo<real,dbl>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << "\n";
   sout << sets << "\n";
   for(int i=0; i<inter.size(); i++)
      sout << inter(i);
   sout << int(simile) << "\n";
   sout << int(endatzero) << "\n";
   sout << int(circular) << "\n";
   sout << int(parallel) << "\n";
   sout << iter << "\n";
   return sout;
   }

// object serialization - loading

template <class real, class dbl> std::istream& turbo<real,dbl>::serialize(std::istream& sin)
   {
   int temp;
   free();
   sin >> encoder;
   sin >> tau;
   sin >> sets;
   inter.init(sets-1);
   for(int i=0; i<inter.size(); i++)
      sin >> inter(i);
   sin >> temp;
   simile = temp != 0;
   sin >> temp;
   endatzero = temp != 0;
   sin >> temp;
   circular = temp != 0;
   sin >> temp;
   parallel = temp != 0;
   sin >> iter;
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

template class turbo<double>;
template <> const serializer turbo<double>::shelper = serializer("codec", "turbo<double>", turbo<double>::create);

template class turbo<mpreal>;
template <> const serializer turbo<mpreal>::shelper = serializer("codec", "turbo<mpreal>", turbo<mpreal>::create);

template class turbo<mpgnu>;
template <> const serializer turbo<mpgnu>::shelper = serializer("codec", "turbo<mpgnu>", turbo<mpgnu>::create);

template class turbo<logreal>;
template <> const serializer turbo<logreal>::shelper = serializer("codec", "turbo<logreal>", turbo<logreal>::create);

template class turbo<logrealfast>;
template <> const serializer turbo<logrealfast>::shelper = serializer("codec", "turbo<logrealfast>", turbo<logrealfast>::create);

template class turbo<logrealfast,logrealfast>;
template <> const serializer turbo<logrealfast,logrealfast>::shelper = serializer("codec", "turbo<logrealfast,logrealfast>", turbo<logrealfast,logrealfast>::create);

}; // end namespace
