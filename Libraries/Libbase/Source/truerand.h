#ifndef __truerand_h
#define __truerand_h

#include "config.h"
#include "vcs.h"

#ifdef WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

/*
  Version 1.00 (20 Apr 2007)
  * created class to provide true random numbers (through OS routines), to facilitate
    seeding slave workers in the master-slave mode
  * idea suggested by Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
  * TODO: currently only returns ival; eventually a 'random' superclass needs to be
    created, which can only be instantiated by deriving and providing the ival
    routine. the new superclass can then provide gaussian and floating point
    numbers by computation
  * TODO: counter also removed, as this should be provided by the superclass

  Version 1.10 (8 May 2007)
  * Win32 support provided through CryptoAPI
  * TODO: when the superclass is created, each generator should only provide an ival()
    function, and the method taking a parameter 'm' should be provided by the superclass.

  Version 1.11 (20 Nov 2007)
  * added error code printing in Win32 when acquiring and releasing CryptoAPI
  * using CRYPT_VERIFYCONTEXT when acquiring CryptoAPI, since we don't need access
    to private keys - this allows use as grid clients, when there is no user profile
    available.
*/

namespace libbase {

class truerand {
   static const vcs version;
#ifdef WIN32
   HCRYPTPROV   hCryptProv;
#else
   int fd;
#endif
public:
   truerand();
   ~truerand();
   inline int32u ival(int32u m);	// return unsigned integer modulo 'm'
};

inline int32u truerand::ival(int32u m)
   {
   int32u x = 0;
#ifdef WIN32
   if(!CryptGenRandom(hCryptProv, sizeof(x), (BYTE *)&x))
#else
   if(read(fd, &x, sizeof(x)) != sizeof(x))
#endif
      {
      std::cerr << "ERROR (truerand): failed to obtain random sequence.\n";
      exit(1);
      }
   return(x % m);
   }

}; // end namespace

#endif
