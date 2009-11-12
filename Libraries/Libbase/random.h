#ifndef __random_h
#define __random_h

#include "config.h"
#include <cmath>
#include <iostream>

namespace libbase {

/*!
 * \brief   Random Generator Base Class.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Defines interface for random generators, and also provides common
 * integer, real (uniform), and Gaussian deviate conversion facility.
 * Implementations of actual random generators are created by deriving
 * from this class and providing the necessary virtual functions.
 */

class random {
private:
   /*! \name Object representation */
#ifndef NDEBUG
   //! Debug only: number of generator advances
   int32u counter;
   //! Debug only: flag to check for explicit seeding
   bool initialized;
#endif
   //! Flag to indicate whether a Gaussian value is readily available
   bool next_gval_available;
   //! Secondary Gaussian value storage
   double next_gval;
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Initialize generator with given seed
   virtual void init(int32u s) = 0;
   //! Advance generator by one step
   virtual void advance() = 0;
   //! The current generator output value
   virtual int32u get_value() const = 0;
   //! The largest returnable value
   virtual int32u get_max() const = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   random();
   virtual ~random();
   // @}

   /*! \name Random generator interface */
   //! Seed random generator
   void seed(int32u s);
   //! Uniformly-distributed unsigned integer in closed interval [0,get_max()]
   int32u ival();
   //! Uniformly-distributed unsigned integer in half-open interval [0,m)
   int32u ival(int32u m)
      {
      assert(m-1 <= get_max());
      return int(floor(fval_halfopen() * m));
      }
   //! Uniformly-distributed floating point value in closed interval [0,1]
   double fval_closed()
      {
      return ival() / double(get_max());
      }
   //! Uniformly-distributed floating point value in half-open interval [0,1)
   double fval_halfopen()
      {
      return ival() / (double(get_max())+1.0);
      }
   //! Return Gaussian-distributed double (zero mean, unit variance)
   double gval();
   //! Return Gaussian-distributed double (zero mean, variance sigma^2)
   double gval(double sigma)
      {
      return gval() * sigma;
      }
   // @}
};

inline int32u random::ival()
   {
#ifndef NDEBUG
   counter++;
   // check for counter roll-over (change to 64-bit counter if this ever happens)
   assert(counter != 0);
   // check for explicit seeding prior to use
   assert(initialized);
#endif
   advance();
   return get_value();
   }

} // end namespace

#endif
