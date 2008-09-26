#ifndef __map_stipple_h
#define __map_stipple_h

#include "map_straight.h"

namespace libcomm {

/*!
   \brief   Stipple Mapper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines an punctured version of the straight mapper, suitable
   for turbo codes, where:
   - all information symbols are transmitted
   - parity symbols are taken from successive sets to result in an overall
     rate of 1/2
   For a two-set turbo code, this corresponds to odd/even puncturing.

   \note This supersedes the puncture_stipple class; observe thought that the
         number of sets here corresponds to the definition used in the turbo
         code, and is one less than that for puncture_stipple.
*/

class map_stipple : public map_straight {
private:
   /*! \name User-defined parameters */
   int tau;    //!< Number of time-steps
   int sets;   //!< Number of turbo code parallel sets
   // @}
   /*! \name Internal object representation */
   libbase::vector<bool> pattern;   //!< Pre-computed puncturing pattern
   // @}
protected:
   /*! \name Internal functions */
   void init(int tau, int sets);
   // @}
   /*! \name Constructors / Destructors */
   map_stipple() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   map_stipple(int tau, int sets) { init(tau, sets); };
   virtual ~map_stipple() {};
   // @}

   // Vector mapper operations
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const;

   // Informative functions
   double rate() const { return (sets+1)/2.0; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_stipple)
};

}; // end namespace

#endif
