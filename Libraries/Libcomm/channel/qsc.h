#ifndef __qsc_h
#define __qsc_h

#include "config.h"
#include "channel.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
 * \brief   q-ary symmetric channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \version 1.00 (12 Feb 2008)
 * - Initial version; implementation of a q-ary symmetric channel as
 * templated class.
 *
 * \version 1.01 (13 Feb 2008)
 * - Fixed check on range of Ps
 * - Changed assert to assertalways in range check
 * - Fixed PDF result for erroneous symbols
 *
 * \version 1.02 (16 Apr 2008)
 * - Fixed computation in corrupt() to force addition within the field
 */

template <class G>
class qsc : public channel<G> {
private:
   /*! \name User-defined parameters */
   double Ps; //!< Symbol-substitution probability \f$ P_s \f$
   // @}
protected:
   // Channel function overrides
   G corrupt(const G& s);
   double pdf(const G& tx, const G& rx) const;
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   qsc()
      {
      }
   // @}

   /*! \name Channel parameter handling */
   //! Set the substitution probability
   void set_parameter(const double Ps);
   //! Get the substitution probability
   double get_parameter() const
      {
      return Ps;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qsc);
};

template <class G>
inline double qsc<G>::pdf(const G& tx, const G& rx) const
   {
   return (tx == rx) ? 1 - Ps : Ps / G::elements();
   }

} // end namespace

#endif

