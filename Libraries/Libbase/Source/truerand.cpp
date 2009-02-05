/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "truerand.h"
#include "stdlib.h"

#ifdef WIN32
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
#ifdef WIN32
   if(!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
      {
      std::cerr << "ERROR (truerand): cannot acquire CryptoAPI context - " << getlasterror() << ".\n";
      exit(1);
      }
#else
   fd = open("/dev/random", O_RDONLY);
   if(fd < 0)
      {
      std::cerr << "ERROR (truerand): cannot open /dev/random.\n";
      exit(1);
      }
#endif
   // call seed to disable check for explicit seeding, since this generator
   // may be used without any seeding at all.
   seed(0);
   }

truerand::~truerand()
   {
#ifdef WIN32
   assert(hCryptProv);
   if(!CryptReleaseContext(hCryptProv, 0))
      {
      std::cerr << "ERROR (truerand): cannot release CryptoAPI context - " << getlasterror() << ".\n";
      exit(1);
      }
#else
   close(fd);
#endif
   }

// Interface with random

inline void truerand::advance()
   {
#ifdef WIN32
   assertalways(CryptGenRandom(hCryptProv, sizeof(x), (BYTE *)&x));
#else
   assertalways(read(fd, &x, sizeof(x)) == sizeof(x));
#endif
   }

}; // end namespace
