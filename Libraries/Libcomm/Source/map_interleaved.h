#ifndef __map_interleaved_h
#define __map_interleaved_h

#include "map_straight.h"
#include "randgen.h"

namespace libcomm {

/*!
   \brief   Mapper Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (28 Apr 2008)
   - Created an interleaved version of the straight mapper.
*/

class map_interleaved : public map_straight {
   libbase::vector<int> lut;
   libbase::randgen r;
public:
   // Vector map_interleaved operations
   void transform(const int N, const libbase::vector<int>& encoded, const int M, libbase::vector<int>& tx);
   void inverse(const libbase::matrix<double>& pin, const int N, libbase::matrix<double>& pout);

   // Setup functions
   void seed(libbase::int32u const s) { r.seed(s); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_interleaved)
};

}; // end namespace

#endif
