#ifndef __map_interleaved_h
#define __map_interleaved_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
   \brief   Mapper Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines an interleaved version of the straight mapper.
*/

class map_interleaved : public map_straight {
   /*! \name Internal object representation */
   libbase::randperm lut;
   libbase::randgen r;
   // @}
public:
   // Vector mapper operations
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out);
   void inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout);

   // Setup functions
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_interleaved)
};

}; // end namespace

#endif
