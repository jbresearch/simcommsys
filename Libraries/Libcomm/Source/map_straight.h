#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Mapper Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (28 Apr 2008)
   - Moved straight symbol mapper from base class.
*/

class map_straight : public mapper {
public:
   // Vector map_straight operations
   void transform(const int N, const libbase::vector<int>& encoded, const int M, libbase::vector<int>& tx);
   void inverse(const libbase::matrix<double>& pin, const int N, libbase::matrix<double>& pout);

   // Informative functions
   double rate() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight)
};

}; // end namespace

#endif
