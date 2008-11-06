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
   int sets;   //!< Number of turbo code parallel sets
   // @}
   /*! \name Internal object representation */
   mutable libbase::vector<bool> pattern;   //!< Pre-computed puncturing pattern
   // @}
protected:
   /*! \name Constructors / Destructors */
   map_stipple() {};
   // @}
   // Interface with mapper
   void advance() const;
   void dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const;
public:
   /*! \name Constructors / Destructors */
   virtual ~map_stipple() {};
   // @}

   // Informative functions
   double rate() const { return (sets+1)/2.0; };
   int output_block_size() const { return tau*2; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_stipple)
};

}; // end namespace

#endif
