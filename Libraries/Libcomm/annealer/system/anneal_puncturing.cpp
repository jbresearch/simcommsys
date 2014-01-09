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

#include "anneal_puncturing.h"
#include <cstdio>
#include <iostream>

namespace libcomm {

anneal_puncturing::anneal_puncturing(const char *fname, const int tau,
      const int s)
   {
   // store user parameters
   anneal_puncturing::tau = tau;
   anneal_puncturing::s = s;
   // initialise contribution matrix and load contribution matrix from file
   contrib.init(s, tau, tau);
   FILE *file = fopen(fname, "rb");
   if (file == NULL)
      {
      std::cerr
            << "FATAL ERROR (anneal_puncturing): Cannot open contribution file ("
            << fname << ")." << std::endl;
      exit(1);
      }
   for (int i = 0; i < tau; i++)
      for (int j = 0; j < s; j++)
         for (int k = 0; k < tau; k++)
            {
            double temp;
            if (fscanf(file, "%lf", &temp) == 0)
               assertalways(fscanf(file, "%*[^\n]\n") == 0);
            contrib(j, i, k) = temp;
            }
   fclose(file);
   // initialise the puncturing pattern as odd-even (and transmitting all data bits)
      {
      pattern.init(s, tau);
      for (int i = 0; i < s; i++)
         for (int j = 0; j < tau; j++)
            pattern(i, j) = (i == 0 || (i - 1) % 2 == j % 2);
      }
   // initialise the working vectors
   res.init(tau);
   res = 0;
   // now work the energy
      {
      for (int i = 0; i < s; i++)
         for (int j = 0; j < tau; j++)
            if (!pattern(i, j))
               energy_function(1, i, j);
      }
   // work out the system's initial energy
   E = work_energy();
   }

anneal_puncturing::~anneal_puncturing()
   {
   output(std::cout);
   }

inline void anneal_puncturing::energy_function(const double factor,
      const int set, const int pos)
   {
   for (int i = 0; i < tau; i++)
      res(i) += factor * contrib(set, pos, i);
   }

double anneal_puncturing::work_energy()
   {
   double sum = 0, sumsq = 0;
   for (int i = 0; i < tau; i++)
      {
      const double x = res(i);
      sum += x;
      sumsq += x * x;
      }
   const double n = tau;
   const double mean = sum / n;
   const double var = sumsq / n - mean * mean;
   return sqrt(var);
   }

double anneal_puncturing::perturb()
   {
   // choose a set and two positions in that set to swap (one must be punctured, the other transmitted)
   set = r.ival(s - 1) + 1;
   do
      {
      pos1 = r.ival(tau);
      } while (pattern(set, pos1));
   do
      {
      pos2 = r.ival(tau);
      } while (!pattern(set, pos2));
   // do the swap
   pattern(set, pos1) = true;
   pattern(set, pos2) = false;
   // and work out the change in energy
   energy_function(1, set, pos1);
   energy_function(-1, set, pos2);
   // update energy
   Eold = E;
   E = work_energy();
   double delta = E - Eold;
   return (delta);
   }

void anneal_puncturing::unperturb()
   {
   // undo the swap
   pattern(set, pos1) = false;
   pattern(set, pos2) = true;
   // and work back the change in energy
   energy_function(-1, set, pos1);
   energy_function(1, set, pos2);
   // revert to old energy value
   E = Eold;
   }

double anneal_puncturing::energy()
   {
   return E;
   }

std::ostream& anneal_puncturing::output(std::ostream& sout) const
   {
   for (int i = 0; i < s; i++)
      {
      sout << pattern(i, 0);
      for (int j = 1; j < tau; j++)
         sout << "\t" << pattern(i, j);
      sout << std::endl;
      }
   return sout;
   }

} // end namespace
