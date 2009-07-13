#ifndef __uniform_lut_h
#define __uniform_lut_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
 \brief   Uniform Interleaver.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 \note This interleaver allows JPL termination.
 */

template <class real>
class uniform_lut : public lut_interleaver<real> {
   libbase::randgen r;
   int tau, m;
protected:
   void init(const int tau, const int m);
   uniform_lut()
      {
      }
public:
   uniform_lut(const int tau, const int m)
      {
      init(tau, m);
      }
   ~uniform_lut()
      {
      }

   void seedfrom(libbase::random& r);
   void advance();

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(uniform_lut);
};

} // end namespace

#endif
