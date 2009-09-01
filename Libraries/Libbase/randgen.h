#ifndef __randgen_h
#define __randgen_h

#include "config.h"
#include "random.h"

namespace libbase {

/*!
 * \brief   Knuth's Subtractive Random Generator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A pseudo-random generator using the subtractive technique due to
 * Knuth. This algorithm was found to give very good results in the
 * communications lab during the third year.
 *
 * \note
 * - The subtractive algorithm has a very long period (necessary for low
 * bit error rates in the tested data stream)
 * - It also does not suffer from low-order correlations (facilitating its
 * use with a variable number of bits/code in the data stream)
 */

class randgen : public random {
private:
   /*! \name Object representation */
   static const int32s mbig;
   static const int32s mseed;
   int32s next, nextp;
   int32s ma[56], mj;
   // @}

protected:
   // Interface with random
   void init(int32u s);
   void advance();
   int32u get_value() const
      {
      return mj;
      }
   int32u get_max() const
      {
      return mbig;
      }
};

} // end namespace

#endif
