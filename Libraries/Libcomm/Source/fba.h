#ifndef __fba_h
#define __fba_h

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
  * fixed include-once definition

  Version 1.21 (5-6 Nov 2007)
  * reduced memory requirements for F and B matrices; instead of catering for all possible
    symbols deleted, we now limit the lowest state to -xmax.
  * fixed error in computing forward and backward metrics: conditions which do not fit the
    given received vector are now skipped (i.e. left at probability zero).
  * fixed error in computing the backward metrics, where we initially needed to access the
    next received bit (ie. beyond the frame).

  Version 1.22 (7 Nov 2007)
  * added debug-mode progress reporting

  Version 1.23 (12 Nov 2007)
  * moved initialization of forward and backward matrices to the functions that compute them
  * removed reset() as it is no longer necessary
  * fixed error in the way backward metrics were handled; the initial condition is the
    drift _after_ transmitting bit t[tau-1], ie before t[tau]. The size of the matrix must
    therefore increase by one, as in the original design.
  * the error fixed in v1.21, where Q() was accessing the next received bit, has now been
    fixed in a more appropriate way, where the backward matrix computation no longer refers
    to bit j+1 but to bit j. TODO: still need to confirm this is right.

  Version 1.24 (14 Nov 2007)
  * optimized work_forward() and work_backward()

  Version 1.25 (15 Nov 2007)
  * further optimized work_backward(), by pre-computing loop limits for y,b
  * ditto for work_forward()
  * added protected getters for I and xmax
  * optimized work_forward() and work_backward() by removing the copying operation on the
    received sequence; this required the provision of sub-vector extraction in vector class

  Version 1.26 (20 Nov 2007)
  * changed references to 'j+1' in Q() as accessed from work_backward() to 'j'.
  * made work_forward() and work_backward() publicly accessible, so that bsid channel can
    avoid computing the backward metrics unnecessarily. Memory allocation checks are now
    done during forward and backward functions.
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
   real& F(const int j, const int y) { return mF(j,y+xmax); };
   real& B(const int j, const int y) { return mB(j,y+xmax); };
   // memory allocation
   void allocate();
protected:
   // getters for parameters
   int get_I() const { return I; };
   int get_xmax() const { return xmax; };
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual dbl P(const int a, const int b) = 0;
   virtual dbl Q(const int a, const int b, const int i, const libbase::vector<sig>& s) = 0;
   // default constructor
   fba();
public:
   // constructor & destructor
   fba(const int tau, const int I, const int xmax);
   virtual ~fba();
   // getters for forward and backward metrics
   real getF(const int j, const int y) const { return mF(j,y+xmax); };
   real getB(const int j, const int y) const { return mB(j,y+xmax); };
   // decode functions
   void work_forward(const libbase::vector<sig>& r);
   void work_backward(const libbase::vector<sig>& r);
   void prepare(const libbase::vector<sig>& r);
   // main initialization routine - constructor essentially just calls this
   void init(const int tau, const int I, const int xmax);
};
   
}; // end namespace

#endif

