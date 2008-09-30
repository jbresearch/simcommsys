#ifndef __lut_interleaver_h
#define __lut_interleaver_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"

namespace libcomm {

/*!
   \brief   Lookup Table Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Document concept of forced tail interleavers (as in divs95)
*/

class lut_interleaver : public interleaver {
protected:
   lut_interleaver() {};
   static const int tail; // a special LUT entry to signify a forced tail
   libbase::vector<int> lut;
public:
   virtual ~lut_interleaver() {};

   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const;
   void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
   void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const;
};

}; // end namespace

#endif

