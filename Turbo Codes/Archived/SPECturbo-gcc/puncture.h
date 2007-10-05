#ifndef __puncture_h
#define __puncture_h

#include "config.h"
#include "vcs.h"

#include "matrix.h"
#include "vector.h"

extern const vcs puncture_version;

/*
  Version 1.00 (7 Jun 1999)
  initial version, abstract class with three implementations (unpunctured, odd/even, and from file).
*/
class puncture {
protected:
   int count, tau, s;
   matrix<int> pattern, pos;
public:
   virtual ~puncture() {};

   int transmit(const int i, const int t) const { return pattern(i, t); };
   int position(const int i, const int t) const { return pos(i, t); };

   int num_symbols() const { return count; };
   double rate() const { return double(count)/double(tau*s); };
   int get_length() const { return tau; };
   int get_sets() const { return s; };

   virtual void print(ostream& s) const = 0;
};

#endif
