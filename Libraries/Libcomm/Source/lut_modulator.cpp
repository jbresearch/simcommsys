#include "lut_modulator.h"

namespace libcomm {

const libbase::vcs lut_modulator::version("LUT Modulator module (lut_modulator)", 1.00);

// modulation/demodulation - atomic operations

const int lut_modulator::demodulate(const sigspace& signal) const
   {
   const int M = map.size();
   int best_i = 0;
   double best_d = signal - map(0);
   for(int i=1; i<M; i++)
      {
      double d = signal - map(i);
      if(d < best_d)
         {
         best_d = d;
         best_i = i;
         }
      }
   return best_i;
   }

// modulation/demodulation - vector operations

void lut_modulator::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx) const
   {
   // Compute factors / sizes & check validity	
   const int M = map.size();
   const int tau = encoded.size();
   const int s = int(libbase::round(log(double(N))/log(double(M))));
   if(N != pow(double(M), s))
      {
      std::cerr << "FATAL ERROR (modulator): each encoder output (" << N << ") must be";
      std::cerr << " represented by an integral number of modulation symbols (" << M << ").";
      std::cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = modulate(x % M);
   }

void lut_modulator::demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int M = map.size();
   // Create a matrix of all possible transmitted symbols
   libbase::matrix<sigspace> tx(1,M);
   for(int x=0; x<M; x++)
      tx(0,x) = modulate(x);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// information functions

double lut_modulator::energy() const
   {
   const int M = map.size();
   double e = 0;
   for(int i=0; i<M; i++)
      e += map(i).r() * map(i).r();
   return e/double(M);
   }

}; // end namespace
