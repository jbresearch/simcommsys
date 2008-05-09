#ifndef __truerand_h
#define __truerand_h

#include "config.h"
#include "random.h"

#ifdef WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif

namespace libbase {

/*!
   \brief   True Random Number Generator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Provide true random numbers (through OS routines), originally created
   to facilitate seeding slave workers in the master-slave mode

   \note
   - idea suggested by Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
   - Win32 support provided through CryptoAPI
   - UNIX support provided through /dev/random
*/

class truerand : public random {
private:
   /*! \name Object representation */
#ifdef WIN32
   HCRYPTPROV   hCryptProv;
#else
   int fd;
#endif
   //! Last generated random value
   int32u x;
   // @}

protected:
   // Interface with random
   void init(int32u s) {};
   void advance();
   int32u get_value() const { return x; };

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   truerand();
   ~truerand();
   // @}

   /*! \name Informative functions */
   //! The largest returnable value
   int32u get_max() const { return 0xffffffff; };
   // @}
};

}; // end namespace

#endif
