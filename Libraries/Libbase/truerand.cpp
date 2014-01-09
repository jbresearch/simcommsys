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

#include "truerand.h"
#include "stdlib.h"

#ifdef _WIN32
#  include <windows.h>
#  include <wincrypt.h>
#else
#  include <iostream>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#endif

namespace libbase {

// Constructors / Destructors

truerand::truerand()
   {
#ifdef _WIN32
   if(!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
      {
      std::cerr << "ERROR (truerand): cannot acquire CryptoAPI context - " << getlasterror() << "." << std::endl;
      exit(1);
      }
#else
   fd = open("/dev/urandom", O_RDONLY);
   if (fd < 0)
      {
      std::cerr << "ERROR (truerand): cannot open /dev/urandom." << std::endl;
      exit(1);
      }
#endif
   // call seed to disable check for explicit seeding, since this generator
   // may be used without any seeding at all.
   seed(0);
   }

truerand::~truerand()
   {
#ifdef _WIN32
   assert(hCryptProv);
   if(!CryptReleaseContext(hCryptProv, 0))
      {
      std::cerr << "ERROR (truerand): cannot release CryptoAPI context - " << getlasterror() << "." << std::endl;
      exit(1);
      }
#else
   close(fd);
#endif
   }

// Interface with random

inline void truerand::advance()
   {
#ifdef _WIN32
   assertalways(CryptGenRandom(hCryptProv, sizeof(x), (BYTE *)&x));
#else
   assertalways(read(fd, &x, sizeof(x)) == sizeof(x));
#endif
   }

} // end namespace
