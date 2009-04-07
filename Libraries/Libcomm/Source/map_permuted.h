#ifndef __map_permuted_h
#define __map_permuted_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
   \brief   Random Symbol Permutation Mapper.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   This class defines an symbol-permuting version of the straight mapper.

   \todo Make this class inherit from any base mapper, not just straight
*/

template <template<class> class C=libbase::vector>
class map_permuted :
   public map_straight<C> {
public:
   /*! \name Type definitions */
   typedef map_straight<C> Base;
   typedef libbase::vector<double>     array1d_t;
   // @}

private:
   /*! \name Internal object representation */
   mutable libbase::vector<libbase::randperm> lut;
   mutable libbase::randgen r;
   // @}

protected:
   // Pull in base class variables
   using Base::M;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

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
