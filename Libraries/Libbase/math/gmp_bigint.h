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

#ifndef __gmp_bigint_h
#define __gmp_bigint_h

#ifdef USE_GMP

#include "config.h"

#include <iostream>
#include <vector>
#include <gmp.h>

namespace libbase {

/*!
 * \brief   BigInteger based on GNU MP.
 * \author  Johann Briffa
 *
 * C++ wrapper for GNU MP integer; provides the following features:
 * - default construction; construction from integer
 * - assignment
 * - random initialization with a given bit length
 * - the number of digits needed to represent the value, given base
 * - pow_mod() method to compute exponentiation modulo m
 * - inv_mod() method to compute inverse modulo m
 * - comparison operators
 * - arithmetic operators: + * %
 * - conversion to/from byte arrays
 * - stream input/output, making use of the dec/oct/hex format specifiers
 */

class gmp_bigint
   {
private:
   static gmp_randstate_t state;
   static bool state_initialized;
   mpz_t value;

public:
   /*! \name Law of the Big Three */
   //! Destructor
   virtual ~gmp_bigint()
      {
      mpz_clear(value);
      }
   //! Copy constructor
   gmp_bigint(const gmp_bigint& x)
      {
      mpz_init_set(value, x.value);
      }
   //! Copy assignment operator
   gmp_bigint& operator=(const gmp_bigint& x)
      {
      mpz_set(value, x.value);
      return *this;
      }
   // @}

   /*! \name Constructors / Destructors */
   //! Default constructor
   explicit gmp_bigint(signed long int x = 0)
      {
      mpz_init_set_si(value, x);
      }
   // @}

   //! Random initialization with a given bit length
   void random(unsigned long bits)
      {
      if(!state_initialized)
         {
         gmp_randinit_default(state);
         state_initialized = true;
         }
      mpz_urandomb(value, state, bits);
      }

   //! The number of digits in the given base, excluding any sign
   size_t size(int base = 2) const
      {
      return mpz_sizeinbase(value, base);
      }

   //! Compute exponentiation modulo m
   gmp_bigint pow_mod(const gmp_bigint& exp, const gmp_bigint& mod) const
      {
      gmp_bigint r;
      mpz_powm(r.value, value, exp.value, mod.value);
      return r;
      }
   //! Compute inverse modulo m
   gmp_bigint inv_mod(const gmp_bigint& mod) const
      {
      gmp_bigint r;
      assertalways(mpz_invert(r.value, value, mod.value) != 0);
      return r;
      }

   /*! \name Comparison operations */
   bool operator==(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) == 0;
      }
   bool operator!=(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) != 0;
      }
   bool operator<=(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) <= 0;
      }
   bool operator>=(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) >= 0;
      }
   bool operator<(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) < 0;
      }
   bool operator>(const gmp_bigint& x) const
      {
      return mpz_cmp(value, x.value) > 0;
      }
   // @}

   /*! \name Arithmetic operations - in-place */
   gmp_bigint& operator+=(const gmp_bigint& x)
      {
      mpz_add(value, value, x.value);
      return *this;
      }
   gmp_bigint& operator*=(const gmp_bigint& x)
      {
      mpz_mul(value, value, x.value);
      return *this;
      }
   gmp_bigint& operator%=(const gmp_bigint& x)
      {
      mpz_mod(value, value, x.value);
      return *this;
      }
   // @}

   /*! \name Arithmetic operations */
   gmp_bigint operator+(const gmp_bigint& x) const
      {
      gmp_bigint r;
      mpz_add(r.value, value, x.value);
      return r;
      }
   gmp_bigint operator*(const gmp_bigint& x) const
      {
      gmp_bigint r;
      mpz_mul(r.value, value, x.value);
      return r;
      }
   gmp_bigint operator%(const gmp_bigint& x) const
      {
      gmp_bigint r;
      mpz_mod(r.value, value, x.value);
      return r;
      }
   // @}

   /*! \name Conversion to/from byte array */
   std::vector<unsigned char> bytearray(bool big_endian = true) const
      {
      // endian-ness flag
      const int order = big_endian ? 1 : -1;
      // determine required size and allocate
      const size_t n = (size() + 8-1) / 8;
      std::vector<unsigned char> v(n);
      // convert and return result
      mpz_export(&v[0], NULL, order, 1, order, 0, value);
      return v;
      }
   explicit gmp_bigint(const std::vector<unsigned char>& v, bool big_endian = true)
      {
      // endian-ness flag
      const int order = big_endian ? 1 : -1;
      // initialize and convert
      mpz_init(value);
      mpz_import(value, v.size(), order, 1, order, 0, &v[0]);
      }
   // TODO: add conversion from byte array
   // @}

   /*! \name Stream I/O */
   friend std::ostream& operator<<(std::ostream& sout, const gmp_bigint& x)
      {
      sout << x.value;
      return sout;
      }

   friend std::istream& operator>>(std::istream& sin, gmp_bigint& x)
      {
      sin >> x.value;
      return sin;
      }
   // @}
   };

} // end namespace

#endif

#endif
