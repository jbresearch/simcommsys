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

#include "cmpi.h"

#include "cputimer.h"
#include "walltimer.h"
#include <iostream>
#ifndef _WIN32
#  include <sys/resource.h>
#endif

namespace libbase {

// Constants

const int cmpi::tag_base = 0;
const int cmpi::tag_data_doublevector = 0x01;
const int cmpi::tag_data_double = 0x02;
const int cmpi::tag_data_int = 0x03;
const int cmpi::tag_getname = 0xFA;
const int cmpi::tag_getusage = 0xFB;
const int cmpi::tag_work = 0xFE;
const int cmpi::tag_die = 0xFF;

const int cmpi::root = 0;

// Static items (initialized to a default value)

bool cmpi::initialized = false;
int cmpi::mpi_rank = -1;
int cmpi::mpi_size = -1;
double cmpi::cpu_usage = 0;

// Static functions

void cmpi::enable(int *argc, char **argv[], const int priority)
   {
   assert(!initialized);
   initialized = true;
#ifdef USE_MPI
   // initialise the MPI routines and determine where we stand
   MPI_Init(argc, argv);
   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
   MPI_Status status;
   // rank zero takes on the job of root/parent process, everone else is a child
   if(mpi_rank > 0)
      {
      // Change the priority information as background task
      const int PRIO_CURRENT = 0;
      setpriority(PRIO_PROCESS, PRIO_CURRENT, priority);
      // get hostname
      int hostname_len = 32;
      char hostname[hostname_len];
      MPI_Get_processor_name(hostname, &hostname_len);
      // Status information for user
      trace << "MPI child system starting on " << hostname << " (priority " << priority << ")." << std::endl;
      // infinite loop, until we are explicitly told to die
      walltimer tim1("MPI latency");
      cputimer tim2("MPI CPU usage");
      tim2.start();
      while(true)
         {
         // Child working function prototype
         void (*func)(void);
         // Get the tag from Root
         tim1.start();
         MPI_Recv(&func, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
         tim1.stop();
         // Status information for user
         trace << "MPI child (rank " << rank() << "/" << size() << ", latency = " << tim1 << "): ";
         // Work out the CPU time used
         double usage = tim2.elapsed();
         // Decide what should be done, based on the tag
         switch(status.MPI_TAG)
            {
            case tag_getname:
            // tell the root process our UNIX hostname
            MPI_Send(hostname, hostname_len, MPI_CHAR, root, tag_base, MPI_COMM_WORLD);
            trace << "send hostname [" << hostname << "]" << std::endl;
            break;
            case tag_getusage:
            // tell the root process how much CPU time we have used
            MPI_Send(&usage, 1, MPI_DOUBLE, root, tag_base, MPI_COMM_WORLD);
            trace << "send usage [CPU " << timer::format(usage) << "]" << std::endl;
            break;
            case tag_work:
            trace << "system working" << std::endl;
            (*func)();
            break;
            case tag_die:
            trace << "system stopped" << std::endl;
            MPI_Finalize();
            exit(0);
            default:
            std::cerr << "received bad tag [" << status.MPI_TAG << "]" << std::endl;
            exit(1);
            }
         }
      }
   // Otherwise, this must be the parent process.
   if(mpi_size > 1)
   trace << "MPI parent system initialized." << std::endl;
   else
      {
      trace << "MPI cluster has one node only - reverting to normal mode." << std::endl;
      initialized = false;
      }
#else
   trace << "MPI class operating in dummy mode - running single." << std::endl;
   initialized = false;
#endif
   }

void cmpi::disable()
   {
   assert(initialized);
#ifdef USE_MPI
   using std::clog;
   // print CPU usage information on the cluster and kill the children
   if(mpi_size > 1)
      {
      MPI_Status status;
      clog << "Processor Usage Summary:" << std::endl;
      cpu_usage = 0;
      for(int i=1; i<mpi_size; i++)
         {
         // now for each node get the number of processors and hostname; finally kill the process
         double usage;
         const int hostname_len = 32;
         char hostname[hostname_len];
         MPI_Send(NULL, 0, MPI_LONG, i, tag_getusage, MPI_COMM_WORLD);
         MPI_Recv(&usage, 1, MPI_DOUBLE, i, tag_base, MPI_COMM_WORLD, &status);
         MPI_Send(NULL, 0, MPI_LONG, i, tag_getname, MPI_COMM_WORLD);
         MPI_Recv(hostname, hostname_len, MPI_CHAR, i, tag_base, MPI_COMM_WORLD, &status);
         MPI_Send(NULL, 0, MPI_LONG, i, tag_die, MPI_COMM_WORLD);
         cpu_usage += usage;
         // inform the user what is happening
         clog << "\tCPU " << i << "\t(" << hostname << "):\t" << timer::format(usage) << std::endl;
         }
      clog << "\tTotal:\t" << timer::format(cpu_usage) << std::endl;
      }
   // cease MPI operations
   MPI_Finalize();
#endif
   initialized = false;
   }

// functions for children to communicate with their parent (the root node)

void cmpi::_receive(double& x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   double a;
   // receive
   MPI_Status status;
   MPI_Recv(&a, 1, MPI_DOUBLE, root, tag_data_double, MPI_COMM_WORLD, &status);
   // fill in results
   x = a;
#endif
   }

void cmpi::_send(const int x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   int a = x;
   // send
   MPI_Send(&a, 1, MPI_INT, root, tag_data_int, MPI_COMM_WORLD);
#endif
   }

void cmpi::_send(const double x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   double a = x;
   // send
   MPI_Send(&a, 1, MPI_DOUBLE, root, tag_data_double, MPI_COMM_WORLD);
#endif
   }

void cmpi::_send(vector<double>& x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   const int count = x.size();
   double a[count];
   // fill in the array
   for(int i=0; i<count; i++)
   a[i] = x(i);
   // send
   MPI_Send(a, count, MPI_DOUBLE, root, tag_data_doublevector, MPI_COMM_WORLD);
#endif
   }

// Non-static items

cmpi::cmpi()
   {
   // only need to do this check here, on creation. If the cmpi object
   // is created, then MPI must be running.
   if (!initialized)
      {
      std::cerr << "FATAL ERROR (cmpi): MPI not initialized." << std::endl;
      exit(1);
      }
   }

cmpi::~cmpi()
   {
   }

// child control function (make a child call a given function)

void cmpi::call(const int rank, void(*func)(void))
   {
#ifdef USE_MPI
   long addr = (long)func;
   MPI_Send(&addr, 1, MPI_LONG, rank+1, tag_work, MPI_COMM_WORLD);
#endif
   }

// functions for communicating with child

void cmpi::receive(int& rank, int& x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   int a;
   // receive
   MPI_Status status;
   MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, tag_data_int, MPI_COMM_WORLD, &status);
   // fill in results
   x = a;
   rank = status.MPI_SOURCE-1;
#endif
   }

void cmpi::receive(int& rank, double& x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   double a;
   // receive
   MPI_Status status;
   MPI_Recv(&a, 1, MPI_DOUBLE, MPI_ANY_SOURCE, tag_data_double, MPI_COMM_WORLD, &status);
   // fill in results
   x = a;
   rank = status.MPI_SOURCE-1;
#endif
   }

void cmpi::receive(int& rank, vector<double>& x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   const int count = x.size();
   double a[count];
   // receive
   MPI_Status status;
   MPI_Recv(a, count, MPI_DOUBLE, MPI_ANY_SOURCE, tag_data_doublevector, MPI_COMM_WORLD, &status);
   // fill in results
   for(int i=0; i<count; i++)
   x(i) = a[i];
   rank = status.MPI_SOURCE-1;
#endif
   }

void cmpi::send(const int rank, const double x)
   {
#ifdef USE_MPI
   // allocate temporary storage
   double a = x;
   // send
   MPI_Send(&a, 1, MPI_DOUBLE, rank+1, tag_data_double, MPI_COMM_WORLD);
#endif
   }

} // end namespace
