#ifndef __nrcc_h
#define __nrcc_h

#include "ccbfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Non-Recursive Convolutional Coder.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class nrcc : public ccbfsm {
protected:
   libbase::vector<int> determineinput(libbase::vector<int> input) const;
   libbase::bitfield determinefeedin(libbase::vector<int> input) const;
   nrcc()
      {
      }
public:
   /*! \name Constructors / Destructors */
   nrcc(libbase::matrix<libbase::bitfield> const &generator) :
      ccbfsm(generator)
      {
      }
   nrcc(nrcc const &x) :
      ccbfsm(x)
      {
      }
   ~nrcc()
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(libbase::vector<int> zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(nrcc);
};

} // end namespace

#endif

