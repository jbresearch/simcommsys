#ifndef __safe_bcjr_h
#define __safe_bcjr_h

#include "bcjr.h"

namespace libcomm {

/*!
 * \brief   Safe version of BCJR - Standard template (no normalization).
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

template <class real, class dbl = double>
class safe_bcjr : public bcjr<real, dbl> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<real, dbl> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<real, dbl> (encoder, tau)
      {
      }
};

/*!
 * \brief   Safe version of BCJR - 'double' specialization (normalized).
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

template <>
class safe_bcjr<double, double> : public bcjr<double, double, true> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<double, double, true> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<double, double, true> (encoder, tau)
      {
      }
};

/*!
 * \brief   Safe version of BCJR - 'float' specialization (normalized).
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

template <>
class safe_bcjr<float, float> : public bcjr<float, float, true> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<float, float, true> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<float, float, true> (encoder, tau)
      {
      }
};

} // end namespace

#endif

