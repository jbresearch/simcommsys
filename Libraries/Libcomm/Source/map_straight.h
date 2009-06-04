#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Straight Mapper Template.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   This class is a template definition for straight mappers; this needs to
   be specialized for actual use. Template parameter defaults are provided
   here.
*/

template <template<class> class C=libbase::vector, class dbl=double>
class map_straight :
   public mapper<C,dbl> {
};

/*!
   \brief   Straight Mapper - Vector containers.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   This class defines a straight symbol mapper with:
   * forward transform from blockmodem
   * inverse transform from the various codecs.
*/

template <class dbl>
class map_straight<libbase::vector,dbl> :
   public mapper<libbase::vector,dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<dbl>        array1d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef mapper<libbase::vector,dbl> Base;
   typedef map_straight<libbase::vector,dbl> This;

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
   void dotransform(const array1i_t& in, array1i_t& out) const;
   void doinverse(const array1vd_t& pin, array1vd_t& pout) const;

public:
   // Informative functions
   double rate() const { return 1; };
   libbase::size<libbase::vector> output_block_size() const
      { return libbase::size<libbase::vector>(size*s1); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight);
};

}; // end namespace

#endif
