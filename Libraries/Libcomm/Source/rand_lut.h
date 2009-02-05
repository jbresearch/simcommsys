#ifndef __rand_lut_h
#define __rand_lut_h

#include "config.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   Random LUT Interleaver.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \note This assumes the implementation of a random simile interleaver; there
         is therefore a restriction that the interleaver size must be a
         multiple of p, where p is the length of the encoder impulse response
         (cf my MPhil p.40). The constructor gives an error if this is not the
         case.
*/

template <class real>
class rand_lut : public lut_interleaver<real> {
   int      p;
   libbase::randgen  r;
protected:
   void init(const int tau, const int m);
   rand_lut() {};
public:
   rand_lut(const int tau, const int m) { init(tau, m); };
   ~rand_lut() {};

   void seedfrom(libbase::random& r);
   void advance();

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(rand_lut);
};

}; // end namespace

#endif
