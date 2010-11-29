#ifndef __fba_h
#define __fba_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"
#include "fsm.h"
#include "multi_array.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Forward-Backward Algorithm.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements Forward-Backward Algorithm for a HMM. This is based on the
 * paper by Davey & McKay, "Watermark Codes: Reliable communication over
 * Insertion/Deletion channels", Trans. IT, 47(2), Feb 2001.
 *
 * Algorithm is implemented on a single block; in the case of Davey's
 * Watermark codes, each block is N elements of n-bit in length, and is the
 * size of the sparsifier's output for a single LDPC codeword. Typical values
 * of n used by Davey were 5,6,7. With watermark codes, N was typically in the
 * range 500-1000. For other examples of LDPC codes, Davey used N up to about
 * 16000.
 *
 * FBA operates at bit-level only, without knowledge of the watermark code
 * and therefore without the possibility of doing the final decode stage.
 *
 * \todo Confirm correctness of the backward matrix computation referring to
 * bit j instead of j+1.
 */

template <class real, class sig, bool norm>
class fba {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef boost::assignable_multi_array<real, 2> array2r_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef fba<real, sig, norm> This;
private:
   /*! \name User-defined parameters */
   int tau; //!< The (transmitted) block size in bits
   int I; //!< The maximum number of insertions per time-step
   int xmax; //!< The maximum allowed drift overall
   double th_inner; //!< Threshold factor for inner cycle
   // @}
   /*! \name Internally-used objects */
   bool initialised; //!< Flag to indicate when memory is allocated
   array2r_t F; //!< Forward recursion metric
   array2r_t B; //!< Backward recursion metric
   // @}
private:
   /*! \name Internal functions */
   void allocate();
   void free();
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real R(const int i, const array1s_t& r) = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba()
      {
      initialised = false;
      }
   virtual ~fba()
      {
      }
   // @}

   // main initialization routine
   void init(int tau, int I, int xmax, double th_inner);
   // getters for forward and backward metrics
   real getF(const int j, const int y) const
      {
      return F[j][y];
      }
   real getB(const int j, const int y) const
      {
      return B[j][y];
      }
   // decode functions
   void work_forward(const array1s_t& r);
   void work_backward(const array1s_t& r);
   void prepare(const array1s_t& r);
};

} // end namespace

#endif
