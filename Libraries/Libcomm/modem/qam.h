#ifndef __qam_h
#define __qam_h

#include "config.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   QAM Modulator.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \version 1.00 (3 Jan 2008)
 * - Initial version, implements square QAM with Gray-coded mapping
 * - Derived from mpsk 2.20
 */

class qam : public lut_modulator {
protected:
   /*! \name Internal operations */
   void init(const int m);
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   qam()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   qam(const int m)
      {
      init(m);
      }
   ~qam()
      {
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qam);
};

} // end namespace

#endif
