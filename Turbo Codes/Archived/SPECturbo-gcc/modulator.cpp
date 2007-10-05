#include "modulator.h"

#include <stdlib.h>

const vcs modulator_version("Modulator Base module (modulator)", 1.00);

const sigspace& modulator::operator[](const int index) const
   {
   if(index < 0 || index >= M)
      {
      cerr << "FATAL ERROR (modulator): Modulation symbol index out of range (" << index << " not in [0," << M-1 << "]\n";
      exit(1);
      }
   return s[index];
   }

const int modulator::operator[](const sigspace& signal) const
   {
   int best_i = 0;
   double best_d = signal - s[0];
   for(int i=1; i<M; i++)
      {
      double d = signal - s[i];
      if(d < best_d)
         {
         best_d = d;
         best_i = i;
         }
      }
   return best_i;
   }

const double modulator::energy() const
   {
   double e = 0;
   for(int i=0; i<M; i++)
      e += s[i].r() * s[i].r();
   return e/double(M);
   }