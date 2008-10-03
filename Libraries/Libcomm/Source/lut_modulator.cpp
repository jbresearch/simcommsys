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

void lut_modulator::domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx)
   {
   // Inherit sizes
   const int M = num_symbols();
   const int tau = this->get_blocksize();
   // Compute factors & check validity
   const int s = int(round( log2(double(N)) / log2(double(M)) ));
   assertalways(tau == encoded.size());
   // Each encoder output N must be representable by an integral number of
   // modulation symbols M
   assertalways(N == pow(M,s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = modulate(x % M);
   }

void lut_modulator::dodemodulate(const channel<sigspace>& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable)
   {
   // Inherit sizes
   const int M = num_symbols();
   const int tau = this->get_blocksize();
   // Check validity
   assertalways(tau == rx.size());
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
