/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "lut_modulator.h"

namespace libcomm {

// modulation/demodulation - atomic operations

const int lut_modulator::demodulate(const sigspace& signal) const
   {
   const int M = lut.size();
   int best_i = 0;
   double best_d = signal - lut(0);
   for(int i=1; i<M; i++)
      {
      double d = signal - lut(i);
      if(d < best_d)
         {
         best_d = d;
         best_i = i;
         }
      }
   return best_i;
   }

// modulation/demodulation - vector operations

void lut_modulator::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx)
   {
   // Compute factors / sizes & check validity  
   const int M = lut.size();
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
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = modulate(x % M);
   }

void lut_modulator::demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable)
   {
   // Compute sizes
   const int M = lut.size();
   // Create a matrix of all possible transmitted symbols
   libbase::vector<sigspace> tx(M);
   for(int x=0; x<M; x++)
      tx(x) = modulate(x);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// information functions

double lut_modulator::energy() const
   {
   const int M = lut.size();
   double e = 0;
   for(int i=0; i<M; i++)
      e += lut(i).r() * lut(i).r();
   return e/double(M);
   }

}; // end namespace
