/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "lut_modulator.h"

namespace libcomm {

// modulation/demodulation - atomic operations

const int lut_modulator::demodulate(const sigspace& signal) const
   {
   const int M = lut.size();
   int best_i = 0;
   double best_d = signal - lut(0);
   for (int i = 1; i < M; i++)
      {
      double d = signal - lut(i);
      if (d < best_d)
         {
         best_d = d;
         best_i = i;
         }
      }
   return best_i;
   }

const int lut_modulator::demodulate(const sigspace& signal, const array1d_t& app) const
   {
   failwith("Method not implemented");
   return 0;
   }

// modulation/demodulation - vector operations

void lut_modulator::domodulate(const int N,
      const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx)
   {
   // Check validity
   assertalways(encoded.size() == this->input_block_size());
   assertalways(N == this->num_symbols());
   // Initialize results vector
   tx.init(this->input_block_size());
   // Inherit sizes
   const int tau = this->input_block_size();
   // Modulate encoded stream
   for (int t = 0; t < tau; t++)
      tx(t) = this->modulate(encoded(t));
   }

void lut_modulator::dodemodulate(const channel<sigspace>& chan,
      const libbase::vector<sigspace>& rx, libbase::vector<array1d_t>& ptable)
   {
   // Check validity
   assertalways(rx.size() == this->input_block_size());
   // Inherit sizes
   const int M = this->num_symbols();
   // Create a matrix of all possible transmitted symbols
   libbase::vector<sigspace> tx(M);
   for (int x = 0; x < M; x++)
      tx(x) = this->modulate(x);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

void lut_modulator::dodemodulate(const channel<sigspace>& chan,
      const libbase::vector<sigspace>& rx,
      const libbase::vector<array1d_t>& app, libbase::vector<array1d_t>& ptable)
   {
   // Do the demodulation step
   dodemodulate(chan, rx, ptable);
   // Factor in prior probabilities
   ptable *= app;
   }

// information functions

double lut_modulator::energy() const
   {
   const int M = lut.size();
   double e = 0;
   for (int i = 0; i < M; i++)
      e += lut(i).r() * lut(i).r();
   return e / double(M);
   }

} // end namespace
