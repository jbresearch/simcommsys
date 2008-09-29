#ifndef __map_interleaved_h
#define __map_interleaved_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
   \brief   Random Interleaving Mapper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines an interleaved version of the straight mapper.
*/

class map_interleaved : public map_straight {
private:
   /*! \name Internal object representation */
   mutable libbase::randperm lut;
   libbase::randgen r;
   // @}

protected:
   // Interface with mapper
   void advance() { lut.init(M,r); };
   void dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void doinverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const;

public:
   // Setup functions
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_interleaved)
};

}; // end namespace

#endif
