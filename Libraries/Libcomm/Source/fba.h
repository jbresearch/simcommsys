#ifndef __fba_h
#define __fba_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"

#include "sigspace.h"
#include "fsm.h"

#include <math.h>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
   \brief   Forward-Backward Algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements Forward-Backward Algorithm for a HMM. This is based on the
   paper by Davey & McKay, "Watermark Codes: Reliable communication over
   Insertion/Deletion channels", Trans. IT, 47(2), Feb 2001.
   
   Algorithm is implemented on a single block; in the case of Davey's
   Watermark codes, each block is N elements of n-bit in length, and is the
   size of the sparsifier's output for a single LDPC codeword. Typical values
   of n used by Davey were 5,6,7. With watermark codes, N was typically in the
   range 500-1000. For other examples of LDPC codes, Davey used N up to about
   16000.

   FBA operates at bit-level only, without knowledge of the watermark code
   and therefore without the possibility of doing the final decode stage.

   \todo Confirm correctness of the backward matrix computation referring to
         bit j instead of j+1.
*/

template <class real, class sig=sigspace>
class fba {
   /*! \name User-defined parameters */
   int   tau;           //!< The (transmitted) block size in bits
   int   I;             //!< The maximum number of insertions considered before every transmission
   int   xmax;          //!< The maximum allowed drift overall
   double th_inner;  //!< Threshold factor for inner cycle
   // @}
   /*! \name Internally-used objects */
   bool  initialised;   //!< Flag to indicate when memory is allocated
   libbase::matrix<real> mF; //!< Forward recursion metric
   libbase::matrix<real> mB; //!< Backward recursion metric
   // @}
private:
   /*! \name Internal functions */
   // index-shifting access internal use
   real& F(const int j, const int y) { return mF(j,y+xmax); };
   real& B(const int j, const int y) { return mB(j,y+xmax); };
   // memory allocation
   void allocate();
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real P(const int a, const int b) = 0;
   virtual real Q(const int a, const int b, const int i, const libbase::vector<sig>& s) = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba() { initialised = false; };
   virtual ~fba() {};
   // @}

   // main initialization routine
   void init(int tau, int I, int xmax, double th_inner);
   // getters for forward and backward metrics
   real getF(const int j, const int y) const { return mF(j,y+xmax); };
   real getB(const int j, const int y) const { return mB(j,y+xmax); };
   // decode functions
   void work_forward(const libbase::vector<sig>& r);
   void work_backward(const libbase::vector<sig>& r);
   void prepare(const libbase::vector<sig>& r);
};

}; // end namespace

#endif

