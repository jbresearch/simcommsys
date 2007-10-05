#ifndef __cmpi_h
#define __cmpi_h

#include "config.h"
#include "vcs.h"

#include "vector.h"

extern const vcs cmpi_version;

class cmpi {
// static items
private:
   static bool initialised;
public:
   static void enable(int *argc, char **argv[], const int priority=10);
   static void disable();
   // informative functions
   static bool enabled() { return initialised; };
   static int nodes() { return 0; };
   static int size() { return 0; };
   static int rank() { return 0; };
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
   void call(const int rank, void (*func)(void));
   // below: receive from any process; rank is updated to the originator
   void receive(int& rank, int& x);
   void receive(int& rank, double& x);
   void receive(int& rank, vector<double>& x);
   void send(const int rank, const double x);
};

#endif
