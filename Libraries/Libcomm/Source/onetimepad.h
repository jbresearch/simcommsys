#ifndef __onetimepad_h
#define __onetimepad_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   One Time Pad Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

template <class real>
class onetimepad : public interleaver<real> {
   bool terminated, renewable;
   fsm *encoder;
   int m, K;
   libbase::vector<int> pad;
   libbase::randgen r;
protected:
   onetimepad();
public:
   onetimepad(const fsm& encoder, const int tau, const bool terminated, const bool renewable);
   onetimepad(const onetimepad& x);
   ~onetimepad();

   void seedfrom(libbase::random& r);
   void advance();

   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;
   void inverse(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(onetimepad);
};

}; // end namespace

#endif
