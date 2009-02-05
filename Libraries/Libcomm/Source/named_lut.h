#ifndef __named_lut_h
#define __named_lut_h

#include "config.h"
#include "lut_interleaver.h"
#include "serializer.h"
#include <string>
#include <iostream>

namespace libcomm {

/*!
   \brief   Named LUT Interleaver.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Implements an interleaver which is specified directly by its LUT,
   and which is externally generated (say by Simulated Annealing
   or another such method).
   A name is associated with the interleaver (say, filename).
*/

template <class real>
class named_lut : public lut_interleaver<real> {
protected:
   std::string lutname;
   int m;
   named_lut() {};
public:
   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(named_lut);
};

}; // end namespace

#endif

