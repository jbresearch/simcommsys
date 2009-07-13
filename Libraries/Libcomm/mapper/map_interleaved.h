#ifndef __map_interleaved_h
#define __map_interleaved_h

#include "map_straight.h"
#include "randperm.h"
#include "randgen.h"

namespace libcomm {

/*!
 \brief   Random Interleaving Mapper.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 This class defines an interleaved version of the straight mapper.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_interleaved : public map_straight<C, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef map_straight<C, dbl> Base;
   typedef map_interleaved<C, dbl> This;

private:
   /*! \name Internal object representation */
   mutable libbase::randperm lut;
   mutable libbase::randgen r;
   // @}

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

public:
   // Setup functions
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_interleaved);
};

} // end namespace

#endif
