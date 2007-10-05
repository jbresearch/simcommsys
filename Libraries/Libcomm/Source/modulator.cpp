#include "modulator.h"
#include "serializer.h"
#include "itfunc.h"
#include <stdlib.h>

namespace libcomm {

const libbase::vcs modulator::version("Modulator Base module (modulator)", 1.40);


// modulation/demodulation - atomic operations

const int modulator::demodulate(const sigspace& signal) const
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

void modulator::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx) const
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

void modulator::demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const
   {
   // Compute sizes
   const int M = map.size();
   const int tau = rx.size();
   // Initialize results vector
   ptable.init(tau, M);
   // Work out the probabilities of each possible signal
   for(int t=0; t<tau; t++)
      for(int x=0; x<M; x++)
         ptable(t,x) = chan.pdf(modulate(x), rx(t));
   }

// information functions

double modulator::energy() const
   {
   const int M = map.size();
   double e = 0;
   for(int i=0; i<M; i++)
      e += map(i).r() * map(i).r();
   return e/double(M);
   }

// serialization functions

std::ostream& operator<<(std::ostream& sout, const modulator* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, modulator*& x)
   {
   std::string name;
   sin >> name;
   x = (modulator*) libbase::serializer::call("modulator", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (modulator): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
