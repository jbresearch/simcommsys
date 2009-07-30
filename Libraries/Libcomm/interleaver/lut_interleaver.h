#ifndef __lut_interleaver_h
#define __lut_interleaver_h

#include "config.h"
#include "interleaver.h"
#include "serializer.h"
#include "fsm.h"

namespace libcomm {

/*!
 * \brief   Lookup Table Interleaver.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * \todo Document concept of forced tail interleavers (as in divs95)
 */

template <class real>
class lut_interleaver : public interleaver<real> {
protected:
   lut_interleaver()
      {
      }
   libbase::vector<int> lut;
public:
   virtual ~lut_interleaver()
      {
      }

   // Transform functions
   void
         transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void
         transform(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;
   void
         inverse(const libbase::matrix<real>& in, libbase::matrix<real>& out) const;

   // Information functions
   int size() const
      {
      return lut.size();
      }
};

} // end namespace

#endif

