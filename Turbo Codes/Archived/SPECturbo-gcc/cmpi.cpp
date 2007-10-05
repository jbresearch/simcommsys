#include "cmpi.h"

#include "timer.h"
#include <iostream.h>

const vcs cmpi_version("Dummy MPI Multi-Computer Environment module (cmpi)", 0.00);

// Static items (initialised to a default value)

bool cmpi::initialised = false;

// Static functions

void cmpi::enable(int *argc, char **argv[], const int priority)
   {
   cerr << "FATAL ERROR (cmpi): MPI not implemented - cannot enable.\n";
   exit(1);
   }

void cmpi::disable()
   {
   }


// functions for children to communicate with their parent (the root node)

void cmpi::_receive(double& x)
   {
   }

void cmpi::_send(const int x)
   {
   }

void cmpi::_send(const double x)
   {
   }

void cmpi::_send(vector<double>& x)
   {
   }

// Non-static items

cmpi::cmpi()
   {
   // only need to do this check here, on creation. If the cmpi object
   // is created, then MPI must be running.
   if(!initialised)
      {
      cerr << "FATAL ERROR (cmpi): MPI not initialised.\n";
      exit(1);
      }
   }

cmpi::~cmpi()
   {
   }

// child control function (make a child call a given function)

void cmpi::call(const int rank, void (*func)(void))
   {
   }

// functions for communicating with child

void cmpi::receive(int& rank, int& x)
   {
   }

void cmpi::receive(int& rank, double& x)
   {
   }

void cmpi::receive(int& rank, vector<double>& x)
   {
   }

void cmpi::send(const int rank, const double x)
   {
   }
