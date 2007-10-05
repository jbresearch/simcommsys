#ifndef __turbo_h
#define __turbo_h

#include "config.h"
#include "vcs.h"

#include "codec.h"
#include "fsm.h"
#include "interleaver.h"
#include "modulator.h"
#include "puncture.h"
#include "channel.h"
#include "bcjr.h"

#include <stdlib.h>
#include <math.h>

#include <fstream.h>

extern const vcs turbo_version;

/*!
\brief   Class implementing the Turbo decoding algorithm.
\author  Johann Briffa
\date    7 June 1999
\version 1.60

  All internal metrics are held as type 'real', which is user-defined. This allows internal working
  at any required level of accuracy. This is required because the internal matrics have a very wide
  dynamic range, which increases exponentially with block size 'tau'. Actually, the required range
  is within [1,0), but very large exponents are required. (For BCJR sub-component)

  Version 1.10 (4 Mar 1999)
  updated intialisation of a priori statistics (now they are 1/K instead of 1).

  Version 1.11 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function (complies with codec 1.01).

  Version 1.20 (5 Mar 1999)
  modified tail_length() to return 0 when the turbo codes is defined with endatzero==false.
  This makes the turbo module handle untailed sequences correctly.

  Version 1.30 (8 Mar 1999)
  modified to use the faster BCJR decode routine (does not compute output statistics).

  Version 1.40 (20 Apr 1999)
  removed the reinitialisation of data probabilities in the tail section for non-simile
  interleavers. The data is always valid input, whether the tails are the same or not.

  Version 1.41 (22 Apr 1999)
  made turbo allocate memory on first call to demodulate/decode

  Version 1.50 (7 Jun 1999)
  modified the system to comply with codec 1.10.

  Version 1.60 (7 Jun 1999)
  revamped handling of puncturing with the use of an additional class.
*/
template <class real> class turbo : public virtual codec, private bcjr<real> {
protected:
   fsm		  *encoder;
   interleaver	  *inter;
   modulator	  *modem;
   puncture       *punc;
   channel	  *chan;
   double         rate;
   int		  tau;
   int            sets;
   bool           simile, endatzero, forceterm;
   int            iter;
   int		  M, K, N, P;    // # of states, inputs, outputs, parity symbols (respectively)
   int		  m;             // memory order of encoder
   int	          S, s;          // # of modulation symbols, # o/p per i/p
   int            sp, sk;        // # of modulation symbols per parity o/p, per i/p
   matrix<double> *R, *r;        // A Priori statistics (intrinsic values)
   matrix<double> ra, rai, raii; // A Priori statistics (extrinsic value)
   matrix<double> ri, rii;	 // A Posteriori statsitics (standard & interleaved)
   // memory allocator (for internal use only)
   bool initialised;             // Initially false, becomes true when memory is initialised
   void allocate();
public:
   turbo(fsm& encoder, modulator& modem, puncture& punc, channel& chan, const int tau, const int sets, interleaver *inter, const int iter, const bool simile, const bool endatzero);
   ~turbo();

   void seed(const int s);

   void encode(vector<int>& source, vector<int>& encoded);
   void modulate(vector<int>& encoded, vector<sigspace>& tx);
   void transmit(vector<sigspace>& tx, vector<sigspace>& rx);

   void demodulate(vector<sigspace>& rx);
   void decode(vector<int>& decoded);
   
   int block_size() { return tau; };
   int num_inputs() { return K; };
   int num_symbols() { return punc->num_symbols(); };
   int tail_length() { return m; };

   int num_iter() { return iter; };
};

template <class real> inline void turbo<real>::seed(const int s)
   {
   for(int set=1; set<sets; set++)
      inter[set-1].seed(s+set);
   }

template <class real> inline turbo<real>::turbo(fsm& encoder, modulator& modem, puncture& punc, channel& chan, const int tau, const int sets, interleaver *inter, const int iter, const bool simile, const bool endatzero) : bcjr<real>(encoder, tau, true, endatzero)
   {
   turbo::encoder = &encoder;
   turbo::modem = &modem;
   turbo::punc = &punc;
   turbo::chan = &chan;
   turbo::tau = tau;
   turbo::sets = sets;
   turbo::inter = inter;
   turbo::simile = simile;
   turbo::endatzero = endatzero;
   turbo::iter = iter;
   
   m = endatzero ? encoder.mem_order() : 0;
   M = encoder.num_states();
   K = encoder.num_inputs();
   N = encoder.num_outputs();
   P = N/K;		// this *must* be an integer
   S = modem.num_symbols();
   sp = int(floor(log(double(P))/log(double(S)) + 0.5));
   sk = int(floor(log(double(K))/log(double(S)) + 0.5));
   s = sk+sets*sp;

   if(P != pow(S, sp))
      {
      cerr << "FATAL ERROR (turbo): Turbo decoder only works when each encoder parity (" << P << ")\n";
      cerr << "   can be represented by an integral number of modulation symbols (" << S << ").\n";
      cerr << "   Suggested number of mod. symbols/encoder parity was " << sp << ".\n";
      exit(1);
      }
   if(K != pow(S, sk))
      {
      cerr << "FATAL ERROR (turbo): Turbo decoder only works when each encoder input (" << K << ")\n";
      cerr << "   can be represented by an integral number of modulation symbols (" << S << ").\n";
      cerr << "   Suggested number of mod. symbols/encoder input was " << sk << ".\n";
      exit(1);
      }

   if(tau != punc.get_length() || s != punc.get_sets())
      {
      cerr << "FATAL ERROR (turbo): Puncturing table size mismatch (" << punc.get_length() << ", " << punc.get_sets() << "), should be (" << tau << ", " << s << ").\n";
      exit(1);
      }
   
   double tN = log2(K*pow(P,sets))*tau * punc.rate();
   double tK = log2(K)*(tau-m);
   rate = tK/tN;
   chan.set_eb(modem.bit_energy()/rate);
   cerr << "Turbo Code initialised, rate = " << rate << "\n";
   cerr << "Modulated with bit energy = " << modem.bit_energy() << "\n";

   // Print information to go into the simulation file
   cout << "#% Codec: Turbo Code (" << tN << "," << tK << ")\n";

   cout << "#% Codec2: ";
   encoder.print(cout);
   cout << ", ";
   inter[0].print(cout);
   cout << ", ";
   cout << (endatzero ? "Terminated, " : "Unterminated, ");
   cout << (simile ? "Simile, " : "Non-simile, ");
   punc.print(cout);
   cout << "\n";

   seed(0);

   initialised = false;
   }

template <class real> inline turbo<real>::~turbo()
   {
   if(initialised)
      {
      delete[] R;
      delete[] r;
      }
   }

template <class real> inline void turbo<real>::allocate()
   {
   R = new matrix<double>[sets];
   r = new matrix<double>[sets];
   for(int i=0; i<sets; i++)
      {
      R[i].init(tau, N);
      r[i].init(tau, K);
      }

   ra.init(tau, K);
   rai.init(tau, K);
   // We only need this intermediate stage when we have more than 2 parallel codes
   if(sets > 2)
      raii.init(tau, K);

   ri.init(tau, K);
   rii.init(tau, K);

   initialised = true;
   }

template <class real> inline void turbo<real>::encode(vector<int>& source, vector<int>& encoded)
   {
   // Advance interleavers to the next block
   for(int set=1; set<sets; set++)
      inter[set-1].advance();

   // Allocate space for the encoder outputs
   matrix<int> x(sets, tau);
   // Allocate space for the interleaved sources
   vector<int> source2(tau);

   // Pass original source through encoder (necessary to know the required tail bits)
   encoder->reset(0);
   for(int t=0; t<tau; t++)
      x(0, t) = encoder->step(source(t)) / K;

   // check that encoder finishes in state zero (applies for all interleaver)
   if(endatzero && encoder->state() != 0)
      cerr << "DEBUG ERROR (turbo): Invalid finishing state for set 0 encoder - " << encoder->state() << "\n";

   // Consider the remaining sets
   for(int set=1; set<sets; set++)
      {
      // Create interleaved source
      inter[set-1].transform(source, source2);
   
      // Encode interleaved source
      encoder->reset(0);
      for(int t=0; t<tau; t++)
         x(set, t) = encoder->step(source2(t)) / K;

      // check that encoder finishes in state zero (applies for all interleaver)
      if(endatzero && encoder->state() != 0)
         cerr << "DEBUG ERROR (turbo): Invalid finishing state for set " << set << " encoder - " << encoder->state() << "\n";
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

template <class real> inline void turbo<real>::modulate(vector<int>& encoded, vector<sigspace>& tx)
   {
   // Pass through modulator
   int x = 0;
   for(int t=0; t<tau; t++)
      for(int i=0, thisx = encoded(t); i<s; i++, thisx /= S)
         if(punc->transmit(i, t))
            tx(x++) = (*modem)[thisx % S];
   }

template <class real> inline void turbo<real>::transmit(vector<sigspace>& tx, vector<sigspace>& rx)
   {
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<num_symbols(); i++)
      rx(i) = chan->corrupt(tx(i));
   }

template <class real> inline void turbo<real>::demodulate(vector<sigspace>& rx)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // Allocate space for temporary matrices
   matrix3<double> p(sets, tau, P);

   // Get the necessary data from the channel
   for(int t=0; t<tau; t++)
      {
      // Input (data) bits [set 0 only]
      for(int x=0; x<K; x++)
         {
         r[0](t, x) = 1;
         for(int i=0, thisx = x; i<sk; i++, thisx /= S)
            if(punc->transmit(i, t))
               r[0](t, x) *= chan->pdf((*modem)[thisx % S], rx(punc->position(i, t)));
         }
      // Parity bits [all sets]
      for(int x=0; x<P; x++)
         for(int set=0, offset=sk; set<sets; set++)
            {
            p(set, t, x) = 1;
            for(int i=0, thisx = x; i<sp; i++, thisx /= S)
               if(punc->transmit(i+offset, t))
                  p(set, t, x) *= chan->pdf((*modem)[thisx % S], rx(punc->position(i+offset, t)));
            offset += sp;
            }
      }

   // Create the interleaved data sets
   for(int set=1; set<sets; set++)
      inter[set-1].transform(r[0], r[set]);

   // Handle tail parity for simile interleavers
   if(simile)
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

   // Initialise a priori probabilities
   for(int t=0; t<tau; t++)
      for(int x=0; x<K; x++)
         ra(t, x) = 1.0/double(K);

   // Compute a priori probabilities (intrinsic)
   for(int set=0; set<sets; set++)
      for(int t=0; t<tau; t++)
         for(int x=0; x<N; x++)
            R[set](t, x) = r[set](t, x%K) * p(set, t, x/K);
   }
   
template <class real> inline void turbo<real>::decode(vector<int>& decoded)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   for(int set=0; set<sets; set++)
      {
      // pass through BCJR algorithm
      bcjr<real>::fdecode(R[set], ra, ri);
      // calculate extrinsic a priori information for next stage
      for(int t=0; t<tau; t++)
         for(int x=0; x<K; x++)
            rai(t, x) = ri(t, x)/(ra(t, x) * r[set](t, x));
      // deinterleave/interleave, ready for next stage
      if(set == 0)
         inter[set].transform(rai, ra);
      else if(set == sets-1)
         inter[set-1].inverse(rai, ra);
      else
         {
         inter[set-1].inverse(rai, raii);
         inter[set].transform(raii, ra);
         }
      }

   // deinterleave the 'full' information as well, for decoding
   inter[sets-2].inverse(ri, rii);

   // Decide which input sequence was most probable, based on BCJR stats.
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(rii(t, i) > rii(t, decoded(t)))
            decoded(t) = i;
      }
   }
   
#endif

