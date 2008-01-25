#ifndef __bsc_h
#define __bsc_h

#include "config.h"
#include "channel.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
   \brief   Binary symmetric channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision: 522 $
   - $Date: 2008-01-24 13:49:24 +0000 (Thu, 24 Jan 2008) $
   - $Author: jabriffa $

   \version 1.00 (25 Jan 2008)
   - Initial version; implementation of a binary symmetric channel.
*/

class bsc : public channel<bool> {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new bsc; };
   // @}
private:
   /*! \name User-defined parameters */
   double   Ps;    //!< Bit-substitution probability \f$ P_s \f$
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s);
   double pdf(const bool& tx, const bool& rx) const;
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   bsc() {};
   // @}
   /*! \name Serialization Support */
   bsc *clone() const { return new bsc(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   /*! \name Channel parameter handling */
   //! Set the substitution probability
   void set_parameter(const double Ps);
   //! Get the substitution probability
   double get_parameter() const { return Ps; };
   // @}

   // Description & Serialization
   std::string description() const;
};

inline double bsc::pdf(const bool& tx, const bool& rx) const
   {
   return (tx != rx) ? Ps : 1-Ps;
   }

}; // end namespace

#endif

