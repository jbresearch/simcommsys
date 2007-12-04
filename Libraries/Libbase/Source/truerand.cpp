#include "truerand.h"

#ifdef WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace libbase {

const vcs truerand::version("True Random Number provider module (truerand)", 1.11);

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

}; // end namespace
