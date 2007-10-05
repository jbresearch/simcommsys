#ifndef __map_h
#define __map_h

#include "config.h"
#include "vcs.h"

#include "codec.h"
#include "fsm.h"
#include "modulator.h"
#include "channel.h"
#include "bcjr.h"
#include "itfunc.h"

#include <stdlib.h>
#include <math.h>

extern const vcs map_version;

/*
  Version 1.01 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function (complies with codec 1.01).

  Version 1.10 (7 Jun 1999)
  modified the system to comply with codec 1.10.
*/
template <class real> class map : public virtual codec, private bcjr<real> {
   fsm	       *encoder;
   modulator   *modem;
   channel     *chan;
   double      rate;
   int	       tau, m;		// block length, and encoder memory order
   int	       M, K, N;		// # of states, inputs and outputs (respectively)
   int	       S, s;		// # of modulation symbols, symbols per encoder o/p
   matrix<double> R, ri, ro;	// BCJR statistics
public:
   map(fsm& encoder, modulator& modem, channel& chan, const int tau);

   void encode(vector<int>& source, vector<int>& encoded);
   void modulate(vector<int>& encoded, vector<sigspace>& tx);
   void transmit(vector<sigspace>& tx, vector<sigspace>& rx);

   void demodulate(vector<sigspace>& rx);
   void decode(vector<int>& decoded);

   int block_size() { return tau; };
   int num_inputs() { return K; };
   int num_symbols() { return tau*s; };
   int tail_length() { return m; };
};

template <class real> inline map<real>::map(fsm& encoder, modulator& modem, channel& chan, const int tau) : bcjr<real>(encoder, tau)
   {
   map::encoder = &encoder;
   map::modem = &modem;
   map::chan = &chan;
   map::tau = tau;
   
   m = encoder.mem_order();
   M = encoder.num_states();
   K = encoder.num_inputs();
   N = encoder.num_outputs();
   S = modem.num_symbols();
   s = int(floor(log(double(N))/log(double(S)) + 0.5));

   if(N != pow(S, s))
      {
      cerr << "FATAL ERROR (map): MAP decoder only works when each encoder output (" << N << ")\n";
      cerr << "   can be represented by an integral number of modulation symbols (" << S << ").\n";
      cerr << "   Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   
   double tN = log2(N)*tau;
   double tK = log2(K)*(tau-m);
   rate = tK/tN;
   chan.set_eb(modem.bit_energy()/rate);
   cerr << "Terminated Convolutional Code initialised, rate = " << rate << "\n";
   cerr << "Modulated with bit energy = " << modem.bit_energy() << "\n";

   // Print information to go into the simulation file
   cout << "#% Codec: Terminated Convolutional Code (" << tN << "," << tK << ")\n";

   cout << "#% Codec2: ";
   encoder.print(cout);
   cout << "\n";


   R.init(tau, N);
   ri.init(tau, K);
   ro.init(tau, N);
   }

template <class real> inline void map<real>::encode(vector<int>& source, vector<int>& encoded)
   {
   // Initialise result vector
   encoded.init(tau);
   // Encode source stream
   encoder->reset(0);
   for(int t=0; t<tau; t++)
      encoded(t) = encoder->step(source(t));
   }

template <class real> inline void map<real>::modulate(vector<int>& encoded, vector<sigspace>& tx)
   {
   // Modulate encoded stream
   for(int t=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, x /= S)
         tx(t*s+i) = (*modem)[x % S];
   }

template <class real> inline void map<real>::transmit(vector<sigspace>& tx, vector<sigspace>& rx)
   {
   // Corrupt the modulation symbols (simulate the channel)
   for(int t=0; t<tau; t++)
      for(int i=0; i<s; i++)
         rx(t*s+i) = chan->corrupt(tx(t*s+i));
   }

template <class real> inline void map<real>::demodulate(vector<sigspace>& rx)
   {
   // Compute the Input statistics for the BCJR Algorithm
   for(int t=0; t<tau; t++)
      for(int x=0; x<N; x++)
         {
         R(t, x) = 1;
         for(int i=0, thisx = x; i<s; i++, thisx /= S)
            R(t, x) *= chan->pdf((*modem)[thisx % S], rx(t*s+i));
         }
   }

template <class real> inline void map<real>::decode(vector<int>& decoded)
   {
   // Decode using BCJR algorithm
   bcjr<real>::decode(R, ri, ro);

   // Decide which input sequence was most probable, based on BCJR stats.
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(ri(t, i) > ri(t, decoded(t)))
            decoded(t) = i;
      }
   }
   
#endif

