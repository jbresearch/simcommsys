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

#ifndef __truerand_h
#define __truerand_h

#include "config.h"
#include "random.h"

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

namespace libbase {

/*!
 * \brief   True Random Number Generator.
 * \author  Johann Briffa
 *
 * Provide "true" random numbers, through OS routines. This was originally
 * created to facilitate seeding slave workers with independent seeds.
 * The random source used is a non-blocking cryptographically secure PRNG,
 * regularly re-seeded from entropy sources available to the kernel.
 * Specifically:
 * - Win32 support provided through CryptoAPI
 * - UNIX support provided through /dev/urandom
 *
 * \note Idea suggested by Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
 */

class truerand : public random {
private:
   /*! \name Object representation */
#ifdef _WIN32
   HCRYPTPROV hCryptProv;
#else
   int fd;
#endif
   //! Last generated random value
   int32u x;
   // @}

protected:
   // Interface with random
   void init(int32u s)
      {
      }
   void advance();
   int32u get_value() const
      {
      return x;
      }
   int32u get_max() const
      {
      return 0xffffffff;
      }

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   truerand();
   ~truerand();
   // @}
};

} // end namespace

#endif
