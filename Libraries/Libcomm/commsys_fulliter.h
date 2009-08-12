#ifndef __commsys_fulliter_h
#define __commsys_fulliter_h

#include "commsys.h"

namespace libcomm {

/*!
 \brief   Communication System with Full-System Iteration.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 Communication system with iterative demodulation and decoding; the model here
 is such that we demodulate once, then decode for M iterations. After that,
 we pass the posterior information as prior information for a second
 demodulation, followed again by M decoding iterations. This is repeated for
 N demodulations (ie. full-system iterations), giving a total of N.M results.

 \todo Integrate this nature within updated commsys interface.
 */

template <class S, template <class > class C = libbase::vector>
class commsys_fulliter : public commsys<S, C> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double> array1d_t;
   // @}

private:
   /*! \name User parameters */
   int iter; //!< Number of demodulation iterations
   // @}
public:
   // Communication System Interface
   void receive_path(const C<S>& received);

   // Description
   std::string description() const;
   // Serialization Support
DECLARE_SERIALIZER(commsys_fulliter);
};

} // end namespace

#endif
