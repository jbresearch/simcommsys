#ifndef __lapgauss_h
#define __lapgauss_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
   \brief   Additive Laplacian-Gaussian Channel.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo this class is still unfinished, and only implements the plain
         Gaussian channel right now
*/

class lapgauss : public channel<sigspace> {
   // channel paremeters
   double               sigma;
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // object handling
   lapgauss();

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(lapgauss);
};

}; // end namespace

#endif

