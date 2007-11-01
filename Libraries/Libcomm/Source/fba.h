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

  Version 1.10 (29-30 Oct 2007)
  * made P() and Q() functions protected (rather then private) and virtual, as these are meant to be defined by derived
    classes. Also made these pure virtual, to ensure they do get defined by a derived class.
  * made the destructor virtual since this class now has virtual functions.
  * added xmax
  * changed definition of received vector to be a vector of signal-space symbols; the actual type
    is a template parameter. This change affects the definitions of decode(), Q(), and word_forward()
    and work_backward().
  * renamed 'n' to 'N' to be consistent with notes & papers.
  * added 'q' to class initialization (this is needed when we come to decode)
  * implemented reset().

  Version 1.20 (31 Oct - 1 Nov 2007)
  * decided to make fba operate at bit-level only, without knowledge of the watermarkcode and
    therefore without the possibility of doing the final decode stage.
    - renamed 'N' to 'tau' to avoid confusion with the symbol-level block size; this is also
      consistent with prior work.
    - removed 'q' as this is no longer necessary.
    - changed decode() to prepare(); made this a protected function as this is meant for use
      by derived classes.
    - renamed F and B matrices and provided protected getters with index-shifting; private
      getters provide index-shifting for internal use. Names had to be different due to an
      ambiguity in use by derived classes.
    - redefined Q() so that the whole received vector is passed (rather than just the last bit)
  * promoted getF, getB, prepare and init to public functions
*/

namespace libcomm {

template <class real, class dbl=double, class sig=sigspace> class fba {
   static const libbase::vcs version;
   // internal variables
   int   tau;  // tau is the (transmitted) block size in bits
   int   I;    // I is the maximum number of insertions considered before every transmission
   int   xmax; // xmax is the maximum allowed drift
   bool  initialised;   // Initially false, becomes true after the first call to "decode" when memory is allocated
   // working matrices
   libbase::matrix<real>   mF; // Forward recursion metric
   libbase::matrix<real>   mB; // Backward recursion metric
   // index-shifting access internal use
   real& F(const int j, const int y) { return mF(j,y+tau-1); };
   real& B(const int j, const int y) { return mB(j,y+tau-1); };
   // memory allocation
   void allocate();
   // internal procedures
   void work_forward(const libbase::vector<sig>& r);
   void work_backward(const libbase::vector<sig>& r);
protected:
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual dbl P(const int a, const int b) = 0;
   virtual dbl Q(const int a, const int b, const int i, const libbase::vector<sig>& s) = 0;
   // reset start- and end-state probabilities
   void reset();
   fba();
public:
   // constructor & destructor
   fba(const int tau, const int I, const int xmax);
   virtual ~fba();
   // getters for forward and backward metrics
   real getF(const int j, const int y) const { return mF(j,y+tau-1); };
   real getB(const int j, const int y) const { return mB(j,y+tau-1); };
   // decode functions
   void prepare(const libbase::vector<sig>& r);
   // main initialization routine - constructor essentially just calls this
   void init(const int tau, const int I, const int xmax);
};
   
}; // end namespace

#endif

