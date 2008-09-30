#ifndef __berrou_h
#define __berrou_h

#include "config.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include "itfunc.h"

namespace libcomm {

/*!
   \brief   Berrou's Original Interleaver.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

template <class real>
class berrou : public lut_interleaver<real> {
   int M;
protected:
   void init(const int M);
   berrou() {};
public:
   berrou(const int M) { init(M); };
   ~berrou() {};

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(berrou)
};

}; // end namespace

#endif

