#ifndef __map_permuted_h
#define __map_permuted_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
   \brief   Random Symbol Permutation Mapper.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines an symbol-permuting version of the straight mapper.
*/

class map_permuted : public map_straight {
private:
   /*! \name Internal object representation */
   mutable libbase::vector<libbase::randperm> lut;
   mutable libbase::randgen r;
   // @}

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   void doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const;

public:
   // Setup functions
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_permuted);
};

}; // end namespace

#endif
