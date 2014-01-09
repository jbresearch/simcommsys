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

#include "anneal_interleaver.h"
#include "vector.h"
#include <cfloat>
#include <algorithm>
#include <iostream>

namespace libcomm {

anneal_interleaver::anneal_interleaver(const int sets, const int tau,
      const int m, const int type, const bool term)
   {
   // store user parameters
   anneal_interleaver::sets = sets;
   anneal_interleaver::tau = tau;
   anneal_interleaver::m = m;
   anneal_interleaver::type = type;
   anneal_interleaver::term = term || (type <= 7);
   // compute useful functions
   f0 = tau * sqrt(double(2));
   f1 = tau * sqrt(double(2)) / 2;
   f2 = pow(0.5 * tau * (tau - 1), 2);
   // initialise LUT and random generator
   lut.init(sets, tau);
   initialise();
   // work out the system's initial energy
   E = work_energy();
   }

void anneal_interleaver::initialise()
   {
   // array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau);
   // period of recursive encoder (assumes feedback polynomial is full)
   const int p = (1 << m) - 1;
   // do the processing loop
   for (int s = 0; s < sets; s++)
      {
      // initialise 'used' status
      used = false;
      // fill in lut
      for (int t = 0; t < tau; t++)
         {
         int tdash;
         do
            {
            tdash = term ? int(r.ival(tau) / p) * p + t % p : r.ival(tau);
            } while (used(tdash));
         used(tdash) = true;
         lut(s, t) = tdash;
         }
      }
   }

inline double anneal_interleaver::energy_function(const int i, const int j)
   {
   if (sets < 2 && type < 15)
      {
      using libbase::PI;

      // compute standard metrics
      const int d_in = abs(j - i);
      const int d_out = abs(lut(0, j) - lut(0, i));
      const double r = sqrt(double(d_in * d_in + d_out * d_out));
      // compute correction metrics
      const double c1 = (r <= tau) ? (r * PI / 2) : (r * (PI / 2 - 2 * acos(tau
            / r)));
      const double c2 = (tau - d_in) * (tau - d_out) / f2;

      // compute change in energy depending on function type
      switch (type)
         {
         case 2:
            return (d_in > 5 * m) ? 0 : (5 * m - d_in + 1)
                  / sqrt(double(d_out));
         case 3:
            return (d_in > 5 * m || d_out > 5 * m) ? 0 : (5 * m - d_in + 1)
                  * (5 * m - d_out + 1);
         case 4:
            return (d_in > 5 * m || d_out > 5 * m) ? 0 : 1;
         case 5:
            return (d_in > 5 * m) ? 0 : 1 / (d_in * d_in * d_out * d_out);
         case 6:
            return (d_in > 5 * m) ? 0 : pow(double(tau - d_in - d_out), 2);
         case 7:
         case 8:
            return 1 / (d_in * d_in * d_out * d_out);
         case 9:
            return (5 * m / r);
         case 10:
            return pow((f0 - r) / f0, 4);
         case 11:
            return (5 * m / r) / c1;
         case 12:
            return pow((r - f1) / f1, 4) / c1;
         case 13:
            return (5 * m / r) / c2;
         case 14:
            {
            using std::min;
            // "count the zeros" approach
            // compute impulse response period
            const int p = (1 << m) - 1;
            // compute length of zeros
            const int l_in = (d_in % p == 0) ? (tau - d_in) : (tau - min(i, j));
            const int l_out = (d_out % p == 0) ? (tau - d_out) : (tau - min(
                  lut(0, i), lut(0, j)));
            // finally compute delta value
            return log(double(l_in + l_out));
            }
         default:
            std::cerr
                  << "FATAL ERROR (anneal_interleaver): Energy function type ["
                  << type << "] for 1 set not implemented" << std::endl;
            exit(1);
         }
      }
   else
      {
      switch (type)
         {
         case 15:
            {
            double d = abs(j - i);
            double sum = d * d;
            for (int s = 0; s < sets; s++)
               {
               d = abs(lut(s, j) - lut(s, i));
               sum += d * d;
               }
            double r = sqrt(sum);
            return (5 * m / r);
            }
         case 16:
            {
            const int d_in = abs(j - i);
            double r = DBL_MAX;
            for (int s = 0; s < sets; s++)
               {
               const int d_out = abs(lut(s, j) - lut(s, i));
               const double r_cur = sqrt(double(d_in * d_in + d_out * d_out));
               if (r > r_cur)
                  r = r_cur;
               }
            return (5 * m / r);
            }
         default:
            std::cerr
                  << "FATAL ERROR (anneal_interleaver): Energy function type ["
                  << type << "] for " << sets << " set(s) not implemented" << std::endl;
            exit(1);
         }
      }

   return 0;
   }

double anneal_interleaver::work_energy()
   {
   double energy = 0;
   for (int i = 0; i < tau; i++)
      for (int j = i + 1; j < tau; j++)
         energy += energy_function(i, j);
   return energy;
   }

double anneal_interleaver::work_delta()
   {
   double delta = 0;
   for (int i = 0; i < tau; i++)
      if (i != pos1 && i != pos2)
         delta += energy_function(pos1, i) + energy_function(pos2, i);
   return delta;
   }

double anneal_interleaver::perturb()
   {
   // randomly choose an interleaver to perturb
   set = r.ival(sets);
   // choose two positions in the lut to swap
   // we restrict pos2>pos1 for minimal random number usage
   if (term)
      {
      const int p = (1 << m) - 1;
      pos1 = r.ival(tau - p);
      pos2 = r.ival(int((tau - 1 - pos1) / p)) * p + pos1;
      }
   else
      {
      pos1 = r.ival(tau - 1);
      pos2 = r.ival(tau - 1 - pos1) + pos1;
      }
   // now work the first part of the change in energy
   double delta = -work_delta();
   // do the swap
   std::swap(lut(set, pos1), lut(set, pos2));
   // and work out the second part of the change in energy
   delta += work_delta();
   // update energy
   Eold = E;
   E += delta;
   return (delta);
   }

void anneal_interleaver::unperturb()
   {
   // swap back the lut values
   std::swap(lut(set, pos1), lut(set, pos2));
   // revert to old energy value
   E = Eold;
   }

double anneal_interleaver::energy()
   {
   return E;
   }

std::ostream& anneal_interleaver::output(std::ostream& sout) const
   {
   for (int s = 0; s < sets; s++)
      {
      sout << "#% Set " << s << std::endl;
      for (int i = 0; i < tau; i++)
         sout << lut(s, i) << std::endl;
      }
   return sout;
   }

} // end namespace
