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

class onetimepad : public interleaver {
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
   void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
   void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(onetimepad)
};

}; // end namespace

#endif
