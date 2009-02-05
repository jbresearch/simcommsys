#ifndef __padded_h
#define __padded_h

#include "config.h"
#include "interleaver.h"
#include "onetimepad.h"
#include "serializer.h"
#include <iostream>

namespace libcomm {

/*!
   \brief   Padded Interleaver.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \note The member onetimepad object is a pointer; this allows us to create
         an empty "padded" class without access to onetimepad's default
         constructor (which is private for that class).
*/

template <class real>
class padded : public interleaver<real> {
   interleaver<real> *otp;
   interleaver<real> *inter;
protected:
   padded();
public:
   padded(const interleaver<real>& inter, const fsm& encoder, const int tau, const bool terminated, const bool renewable);
   padded(const padded& x);
   ~padded();

   void seedfrom(libbase::random& r);
   void advance();

   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;
   void inverse(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(padded);
};

}; // end namespace

#endif
