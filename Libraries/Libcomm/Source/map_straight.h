#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Straight Mapper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines a straight symbol mapper with:
   * forward transform from modulator
   * inverse transform from the various codecs.
*/

class map_straight : public mapper {
private:
   /*! \name Internal object representation */
   int s1;  //!< Number of modulation symbols per encoder output
   int s2;  //!< Number of modulation symbols per translation symbol
   // @}

   // Interface with derived classes
   void setup();

public:
   // Vector mapper operations
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const;

   // Informative functions
   double rate() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight)
};

}; // end namespace

#endif
