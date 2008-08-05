#ifndef __parametric_h
#define __parametric_h

#include "config.h"

namespace libcomm {

/*!
   \brief   Parametric Clas Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Defines a class that takes a scalar parameter.
*/

class parametric {
public:
   /*! \name Constructors / Destructors */
   virtual ~parametric() {};
   // @}

   /*! \name Parameter handling */
   //! Set the characteristic parameter
   virtual void set_parameter(const double x) = 0;
   //! Get the characteristic parameter
   virtual double get_parameter() const = 0;
   // @}

};

}; // end namespace

#endif
