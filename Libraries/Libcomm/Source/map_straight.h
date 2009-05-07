#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Straight Mapper.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   This class defines a straight symbol mapper with:
   * forward transform from blockmodem
   * inverse transform from the various codecs.

   \bug This is really only properly defined for vector containers.
*/

template <template<class> class C=libbase::vector, class dbl=double>
class map_straight :
   public mapper<C,dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl>     array1d_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef mapper<C,dbl> Base;
   typedef map_straight<C,dbl> This;

private:
   /*! \name Internal object representation */
   int s1;        //!< Number of modulation symbols per encoder output
   int s2;        //!< Number of modulation symbols per translation symbol
   int upsilon;   //!< Block size in symbols at codec translation
   // @}

protected:
   // Pull in base class variables
   using Base::size;
   using Base::M;
   using Base::N;
   using Base::S;

protected:
   // Interface with mapper
   void setup();
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

public:
   // Informative functions
   double rate() const { return 1; };
   libbase::size<C> output_block_size() const { return libbase::size<C>(size*s1); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight);
};

}; // end namespace

#endif
