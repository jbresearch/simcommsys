#ifndef __interleaver_h
#define __interleaver_h

#include "config.h"
#include "vcs.h"

#include "matrix.h"
#include "vector.h"

extern const vcs interleaver_version;

class interleaver {
public:
   virtual void seed(const int s) {};
   virtual void advance() {};
   // note that 'in' and 'out' may NOT be the same!
   virtual void transform(vector<int>& in, vector<int>& out) const = 0;
   virtual void transform(matrix<double>& in, matrix<double>& out) const = 0;
   virtual void inverse(matrix<double>& in, matrix<double>& out) const = 0;

   virtual void print(ostream& s) const = 0;
};

#endif

