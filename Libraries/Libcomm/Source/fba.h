#ifndef __bcjr_h
#define __bcjr_h

#include "config.h"
#include "vcs.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"

#include "sigspace.h"
#include "fsm.h"

#include <math.h>
#include <iostream>
#include <fstream>

/*
  Version 1.00 (1-11 Oct 2007)
  * Initial version, implementing Forward-Backward Algorithm for a HMM. This is based on the paper by Davey & McKay,
    "Watermark Codes: Reliable communication over Insertion/Deletion channels", Trans. IT, 47(2), Feb 2001.
  * Implements algorithm on a single block; in the case of Davey's Watermark codes, each block is N elements of n-bit in
    length, and is the size of the sparsifier's output for a single LDPC codeword. Typical values of n were 5,6,7. With
    watermark codes, N was typically in the range 500-1000. For other examples of LDPC codes, Davey used N up to
    about 16000.

  Version 1.10 (29 Oct 2007)
  * made P() and Q() functions protected (rather then private) and virtual, as these are meant to be defined by derived
    classes. Also made these pure virtual, to ensure they do get defined by a derived class.
  * made the destructor virtual since this class now has virtual functions.
  * added xmax
*/

namespace libcomm {

template <class real, class dbl=double> class fba {
   static const libbase::vcs version;
   // internal variables
   int   n;    // n is the size of the block, on the input side
   int   I;    // I is the maximum number of insertions considered before every transmission
   int   xmax; // xmax is the maximum allowed drift
   bool  initialised;   // Initially false, becomes true after the first call to "decode" when memory is allocated
   // working matrices
   libbase::matrix<real>   F; // Forward recursion metric
   libbase::matrix<real>   B; // Backward recursion metric
   // memory allocation
   void allocate();
   // internal procedures
   void work_forward(const libbase::vector<int>& r);
   void work_backward(const libbase::vector<int>& r);
protected:
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual dbl P(const int a, const int b) = 0;
   virtual dbl Q(const int a, const int b, const int i, const int s) = 0;
   // main initialization routine - constructor essentially just calls this
   void init(const int n, const int I, const int xmax);
   // reset start- and end-state probabilities
   void reset();
   fba();
public:
   // constructor & destructor
   fba(const int n, const int I, const int xmax);
   virtual ~fba();
   // decode functions
   void decode(const libbase::vector<int>& r, libbase::matrix<dbl>& p);
};
   
}; // end namespace

#endif

