#ifndef __sysrepacc_h
#define __sysrepacc_h

#include "config.h"
#include "repacc.h"
#include "codec_softout_flattened.h"

namespace libcomm {

/*!
   \brief   Systematic Repeat-Accumulate (SRA) codes.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Extension of the Repeat-Accumulate (RA) codes, also transmitting
   systematic data on the channel.
*/

template <class real, class dbl=double>
class sysrepacc : public codec_softout_flattened< repacc<real,dbl>, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<dbl>        array1d_t;
   typedef libbase::matrix<dbl>        array2d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef sysrepacc<real,dbl> This;
   typedef codec_softout_flattened< repacc<real,dbl>, dbl> Base;
   typedef bcjr<real,dbl> BCJR;
   // Grant access to inherited fields and methods
   using Base::ra;
   using Base::rp;
   using Base::R;
   using Base::initialised;
   using Base::allocate;
   using Base::reset;
public:
   /*! \name Constructors / Destructors */
   ~sysrepacc() {};
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const libbase::vector< libbase::vector<double> >& ptable);

   // Codec information functions - fundamental
   libbase::size<libbase::vector> output_block_size() const
      { return libbase::size<libbase::vector>(Base::input_block_size() + Base::output_block_size()); };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(sysrepacc);
};

}; // end namespace

#endif
