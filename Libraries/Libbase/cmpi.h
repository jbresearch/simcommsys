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

#ifndef __cmpi_h
#define __cmpi_h

#include "config.h"
#include "vector.h"

#ifdef USE_MPI
#  include <mpi.h>
#endif

namespace libbase {

/*!
 * \brief   MPI Multi-Computer Environment.
 * \author  Johann Briffa
 *
 * \version 1.01 (5 Mar 1999)
 * updated mpi child functions to print the latency between jobs.
 *
 * \version 2.00 (23 Mar 1999)
 * major update to the class.
 *
 * \version 2.10 (25 Mar 1999)
 * changed the spawning algorithm. Now, this will spawn max*nodes processes, so that all processors
 * are necessarily covered. Then, it will kill unnecessary nodes to avoid memory wastage. Note that
 * with this technique, there is no more the need to have the nodes sorted in order of decreasing
 * processors. Also, all nodes are cleanly in a single communicator, which aviods playing about with
 * MPI. Also changed the initialisation process of children. Now children enter the loop immediately
 * and must explicitly directed to send the number of processors to root or to get the size/rank.
 *
 * \version 2.11 (7 May 1999)
 * re-schedules children with a lower priority to favour foreground tasks.
 *
 * \version 2.12 (22 Jul 1999)
 * prints CPU usage information at the end of the process.
 *
 * \version 2.13 (3 Sep 1999)
 * added functions for children to send integers to the root process
 *
 * \version 2.20 (29 Sep 2001)
 * modified to compile on Win32 as a Dummy MPI module
 *
 * \version 2.21 (23 Feb 2002)
 * added flushes to all end-of-line clog outputs, to clean up text user interface.
 *
 * \version 2.22 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 2.30 (26 Jul 2006)
 * first attempt to get this to work with OpenMPI as installed on iapetus (should be
 * a recent version as it has just been installed for me, but I do not yet know how
 * to determine the version number).
 * - made as much code unconditional as possible (instead of depending on the
 * definition of MPI, which was conflicting with the namespace)
 * - reworked the parent/child relationship to avoid spawning - modern-day libaries
 * seem to handle multi-CPU machines with ease, and this greatly simplifies the code.
 * - removed dependence on environment variables 'CPUS' and 'HOSTNAME'
 *
 * \version 2.31 (27 Jul 2006)
 * updated class:
 * - to revert to non-MPI mode when the "cluster" consists of a single node.
 * - to alleviate typos, the 'initialised' variable has been americanized.
 * - the 'rank()' function is now also scaled down to ignore the root process.
 * - CPU usage is returned as type double instead of integer (for each child)
 * - total usage is kept as a private member and updated at the end; this can be
 * read by the user with the appropriate function.
 *
 * \version 2.40 (26 Oct 2006)
 * - defined class and associated data within "libbase" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 *
 * \version 2.41 (28 Sep 2007)
 * modified so that compilation as a Dummy MPI module occurs when USEMPI is not defined.
 */

class cmpi {

   // constants
private:
   static const int tag_base;
   static const int tag_data_doublevector;
   static const int tag_data_double;
   static const int tag_data_int;
   static const int tag_getname;
   static const int tag_getusage;
   static const int tag_work;
   static const int tag_die;
public:
   static const int root;

   // static items
private:
   static bool initialized;
   static int mpi_rank, mpi_size;
   static double cpu_usage;
public:
   static void enable(int *argc, char **argv[], const int priority = 10);
   static void disable();
   // informative functions
   static bool enabled()
      {
      return initialized;
      }
   // the two values below are scaled to ignore the root process
   static int size()
      {
      return mpi_size - 1;
      }
   static int rank()
      {
      return mpi_rank - 1;
      }
   // return total CPU usage for all child processes
   static double usage()
      {
      return cpu_usage;
      }
   // parent communication functions
   static void _receive(double& x);
   static void _send(const int x);
   static void _send(const double x);
   static void _send(vector<double>& x);

   // non-static items
public:
   // creation and destruction
   cmpi();
   ~cmpi();
   // child control functions
   void call(const int rank, void(*func)(void));
   // below: receive from any process; rank is updated to the originator
   void receive(int& rank, int& x);
   void receive(int& rank, double& x);
   void receive(int& rank, vector<double>& x);
   void send(const int rank, const double x);
};

} // end namespace

#endif
