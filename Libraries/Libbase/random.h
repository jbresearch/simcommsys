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
#include <cstdint>
#include <iostream>

namespace libbase
{

// Determine debug level:
// 1 - Normal debug output only
// 2 - Track construction/destruction
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
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

class random
{
private:
    /*! \name Object representation */
#ifndef NDEBUG
    //! Debug only: number of generator advances
    uint32_t counter;
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
#ifdef __CUDACC__
    __device__
    __host__
#endif
    virtual void init(uint32_t s) = 0;
//! Advance generator by one step
#ifdef __CUDACC__
    __device__
    __host__
#endif
    virtual void advance() = 0;
//! The current generator output value
#ifdef __CUDACC__
    __device__
    __host__
#endif
    virtual uint32_t get_value() const = 0;
//! The largest returnable value
#ifdef __CUDACC__
    __device__
    __host__
#endif
    virtual uint32_t get_max() const = 0;
    // @}

public:
/*! \name Constructors / Destructors */
//! Principal constructor
#ifdef __CUDACC__
    __device__
    __host__
#endif
    random()
    {
#ifndef NDEBUG
        counter = 0;
        initialized = false;
#endif
#if DEBUG >= 2
        std::cerr << "DEBUG: random (" << this << ") created." << std::endl;
#endif
        next_gval_available = false;
    }
//! Copy constructor
#ifdef __CUDACC__
    __device__
    __host__
#endif
    random(const random& r)
        :
#ifndef NDEBUG
          counter(r.counter), initialized(r.initialized),
#endif
          next_gval_available(r.next_gval_available), next_gval(r.next_gval)
    {
#if DEBUG >= 2
        std::cerr << "DEBUG: random (" << this << ") created as a copy of ("
                  << &r << ")." << std::endl;
#endif
    }
//! Copy assignment
#ifdef __CUDACC__
    __device__
    __host__
#endif
    random& operator=(const random& r)
    {
#ifndef NDEBUG
        counter = r.counter;
        initialized = r.initialized;
#endif
        next_gval_available = r.next_gval_available;
        next_gval = r.next_gval;
#if DEBUG >= 2
        std::cerr << "DEBUG: random (" << this << ") copied from (" << &r
                  << ")." << std::endl;
#endif
        return *this;
    }
//! Virtual destructor
#ifdef __CUDACC__
    __device__
    __host__
#endif
    virtual ~random()
    {
#if DEBUG >= 2
        std::cerr << "DEBUG: random (" << this << ") destroyed after "
                  << counter << " steps." << std::endl;
#endif
    }
// @}

/*! \name Random generator interface */
//! Seed random generator
#ifdef __CUDACC__
    __device__
    __host__
#endif
    void seed(uint32_t s)
    {
#if DEBUG >= 2
        std::cerr << "DEBUG: random (" << this << ") reseeded with " << s
                  << " after " << counter << " steps." << std::endl;
#endif
#ifndef NDEBUG
        counter = 0;
        initialized = true;
#endif
        // this makes sure any stored gval is discarded
        next_gval_available = false;
        // initialize underlying generator
        init(s);
    }

//! Uniformly-distributed unsigned integer in closed interval [0,get_max()]
#ifdef __CUDACC__
    __device__
    __host__
#endif
    uint32_t ival()
    {
#ifndef NDEBUG
        counter++;
        // check for counter roll-over (change to 64-bit counter if this ever
        // happens)
        assert(counter != 0);
        // check for explicit seeding prior to use
        assert(initialized);
#endif
        advance();
        return get_value();
    }
//! Uniformly-distributed unsigned integer in half-open interval [0,m)
#ifdef __CUDACC__
    __device__
    __host__
#endif
    uint32_t ival(uint32_t m)
    {
        assert(m - 1 <= get_max());
        return int(floor(fval_halfopen() * m));
    }
//! Uniformly-distributed floating point value in closed interval [0,1]
#ifdef __CUDACC__
    __device__
    __host__
#endif
    double fval_closed() { return ival() / double(get_max()); }
//! Uniformly-distributed floating point value in half-open interval [0,1)
#ifdef __CUDACC__
    __device__
    __host__
#endif
    double fval_halfopen() { return ival() / (double(get_max()) + 1.0); }
//! Return Gaussian-distributed double (zero mean, unit variance)
#ifdef __CUDACC__
    __device__
    __host__
#endif
    double gval()
    {
        if (next_gval_available) {
            next_gval_available = false;
            return next_gval;
        }

        double v1, v2, rsq;
        do {
            v1 = 2.0 * fval_closed() - 1.0;
            v2 = 2.0 * fval_closed() - 1.0;
            rsq = (v1 * v1) + (v2 * v2);
        } while (rsq >= 1.0 || rsq == 0.0);
        double fac = sqrt(-2.0 * log(rsq) / rsq);
        next_gval = v2 * fac;
        next_gval_available = true;
        return (v1 * fac);
    }
//! Return Gaussian-distributed double (zero mean, variance sigma^2)
#ifdef __CUDACC__
    __device__
    __host__
#endif
    double gval(double sigma) { return gval() * sigma; }
    // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG
#endif

} // namespace libbase

#endif
