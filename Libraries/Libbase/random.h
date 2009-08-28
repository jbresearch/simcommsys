#ifndef __random_h
#define __random_h

#include "config.h"
#include <math.h>
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
 * integer, real (uniform), and gaussian deviate conversion facility.
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
   //! Return unsigned integer in [0, getmax()]
   int32u ival();
   //! Return unsigned integer modulo 'm'
   int32u ival(int32u m)
      {
      assert(m-1 <= get_max());
      return ival() % m;
      }
   //! Return floating point value in closed interval [0,1]
   double fval()
      {
      return ival() / double(get_max());
      }
   //! Return gaussian-distributed double (zero mean, unit variance)
   double gval();
   //! Return gaussian-distributed double (zero mean, variance sigma^2)
   double gval(double sigma)
      {
      return gval() * sigma;
      }
   // @}

   /*! \name Informative functions */
   //! The largest returnable value
   virtual int32u get_max() const = 0;
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
