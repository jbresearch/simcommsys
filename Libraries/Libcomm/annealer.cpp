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

#include "annealer.h"
#include <cmath>
#include <iostream>

namespace libcomm {

void annealer::attach_system(anneal_system& system)
   {
   annealer::system = &system;
   // Set default parameters
   double E = pow(double(10), double(ceil(log10(system.energy()))));
   set_temperature(E, E * 1E-7);
   set_schedule(0.90);
   set_iterations(int(1E5), int(1E3));
   }

void annealer::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   system->seedfrom(r);
   }

void annealer::set_temperature(const double Tstart, const double Tstop)
   {
   annealer::Tstart = Tstart;
   annealer::Tstop = Tstop;
   }

void annealer::set_schedule(const double rate)
   {
   annealer::rate = rate;
   }

void annealer::set_iterations(const int min_iter, const int min_changes)
   {
   annealer::min_iter = min_iter;
   annealer::min_changes = min_changes;
   }

void annealer::improve()
   {
   // set stderr to precision 4
   std::streamsize prec = std::clog.precision(4);

   // 'E' holds the instantaneous system energy
   // 'stability' denotes the number of successive steps for which system did not change
   double E = system->energy();
   int stability = 0;
   for (double T = Tstart; T > Tstop && stability < 5; T *= rate)
      {
      // limits to be plotted at this temperature
      libbase::rvstatistics stat;
      // main loop - repeated until number of iterations >= min_iter,
      //             or number of system state changes >= min_changes,
      //             whichever comes first
      int i, c;
      for (i = 0, c = 0; i < min_iter && c < min_changes; i++)
         {
         double deltaE = system->perturb();
         if (deltaE < 0 || r.fval_closed() < exp(-deltaE / T))
            {
            c++;
            E += deltaE;
            stat.insert(E);
            }
         else
            system->unperturb();
         if (interrupt())
            return;
         }
      // check for stability (count *consecutive* runs of zero-change iterations)
      if (c == 0)
         stability++;
      else
         stability = 0;
      // output some statistics for the user
      display(T, 100 * double(c) / double(i), stat);
      }

   // revert stderr to original precision
   std::clog.precision(prec);
   }

void annealer::display(const double T, const double percent,
      const libbase::rvstatistics E)
   {
   std::clog << T << "\t" << E.hi() << "\t" << E.lo() << "\t" << percent
         << "%" << std::endl;
   }

} // end namespace
