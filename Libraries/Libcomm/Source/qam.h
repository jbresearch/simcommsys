#ifndef __qam_h
#define __qam_h

#include "config.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   QAM Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (3 Jan 2008)
   - Initial version, implements square QAM with Gray-coded mapping
   - Derived from mpsk 2.20
*/

class qam : public lut_modulator {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new qam; };
   // @}
protected:
   /*! \name Internal operations */
   void init(const int m);
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   qam() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   qam(const int m) { init(m); };
   ~qam() {};
   // @}

   /*! \name Class management (cloning/naming) */
   qam *clone() const { return new qam(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   /*! \name Description & Serialization */
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
   // @}
};

}; // end namespace

#endif
