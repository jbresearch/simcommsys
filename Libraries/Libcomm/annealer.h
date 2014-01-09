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

#ifndef __annealer_h
#define __annealer_h

#include "config.h"
#include "annealer/anneal_system.h"
#include "randgen.h"
#include "rvstatistics.h"

namespace libcomm {

/*!
 * \brief   Simulated Annealing Algorithm.
 * \author  Johann Briffa
 *
 * \version 1.01 (10 Oct 2001)
 * added a virtual display function, to facilitate deriving from the class to produce a
 * windowed GUI (by using custom display routines), and also added a virtual interrupt
 * function to allow a derived class to stop the processing routine. Both functions are
 * protected so they can only be called by the class itself or by derived classes.
 *
 * \version 1.02 (16 Nov 2001)
 * added a virtual destructor.
 *
 * \version 1.03 (23 Feb 2002)
 * added flushes to all end-of-line clog outputs, to clean up text user interface.
 *
 * \version 1.04 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.10 (27 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class annealer {
protected:
   anneal_system *system;
   libbase::randgen r;
   double Tstart, Tstop, rate;
   int min_iter, min_changes;
protected:
   virtual ~annealer()
      {
      }
   virtual bool interrupt()
      {
      return false;
      }
   virtual void display(const double T, const double percent,
         const libbase::rvstatistics E);
public:
   void attach_system(anneal_system& system);
   void seedfrom(libbase::random& r);
   void set_temperature(const double Tstart, const double Tstop);
   void set_schedule(const double rate);
   void set_iterations(const int min_iter, const int min_changes);
   void improve();
};

} // end namespace

#endif
