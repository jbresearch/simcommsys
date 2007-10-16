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
*/

namespace libcomm {

template <class real, class dbl=double> class fba {
   static const libbase::vcs version;
   // internal variables
   int   n;    // n is the size of the block, on the input side
   int   I;    // I is the (artificial) maximum number of insertions before every transmission
   bool  initialised;   // Initially false, becomes true after the first call to "decode" when memory is allocated
   // working matrices
   libbase::matrix<real>   F; // Forward recursion metric
   libbase::matrix<real>   B; // Backward recursion metric
   // memory allocation
   void allocate();
   // internal procedures
   void work_forward(const libbase::vector<int>& r);
   void work_backward(const libbase::vector<int>& r);
   // handles for channel-specific metrics - to be implemented by derived classes
   dbl P(const int a, const int b) { return 0; };
   dbl Q(const int a, const int b, const int i, const int s) { return 0; };
protected:
   // main initialization routine - constructor essentially just calls this
   void init(const int n, const int I);
   // reset start- and end-state probabilities
   void reset();
   fba();
public:
   // constructor & destructor
   fba(const int n, const int I);
   ~fba();
   // decode functions
   void decode(const libbase::vector<int>& r, libbase::matrix<dbl>& p);
};
   
}; // end namespace

#endif

