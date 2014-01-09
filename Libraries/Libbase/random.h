/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __random_h
#define __random_h

#include "config.h"
#include <cmath>
#include <iostream>

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Track construction/destruction
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Random Generator Base Class.
 * \author  Johann Briffa
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
   random()
      {
#ifndef NDEBUG
      counter = 0;
      initialized = false;
#endif
#if DEBUG>=2
      std::cerr << "DEBUG: random (" << this << ") created." << std::endl;
#endif
      next_gval_available = false;
      }
   //! Copy constructor
   random(const random& r) :
#ifndef NDEBUG
            counter(r.counter), initialized(r.initialized),
#endif
            next_gval_available(r.next_gval_available), next_gval(r.next_gval)
      {
#if DEBUG>=2
      std::cerr << "DEBUG: random (" << this << ") created as a copy of ("
            << &r << ")." << std::endl;
#endif
      }
   //! Copy assignment
   random& operator=(const random& r)
      {
#ifndef NDEBUG
      counter = r.counter;
      initialized = r.initialized;
#endif
      next_gval_available = r.next_gval_available;
      next_gval = r.next_gval;
#if DEBUG>=2
      std::cerr << "DEBUG: random (" << this << ") copied from (" << &r << ")."
            << std::endl;
#endif
      return *this;
      }
   //! Virtual destructor
   virtual ~random()
      {
#if DEBUG>=2
      std::cerr << "DEBUG: random (" << this << ") destroyed after " << counter
            << " steps." << std::endl;
#endif
      }
   // @}

   /*! \name Random generator interface */
   //! Seed random generator
   void seed(int32u s);
   //! Uniformly-distributed unsigned integer in closed interval [0,get_max()]
   int32u ival()
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
   //! Uniformly-distributed unsigned integer in half-open interval [0,m)
   int32u ival(int32u m)
      {
      assert(m - 1 <= get_max());
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
      return ival() / (double(get_max()) + 1.0);
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

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
