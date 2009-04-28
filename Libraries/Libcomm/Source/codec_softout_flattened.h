#ifndef __codec_softout_flattened_h
#define __codec_softout_flattened_h

#include "config.h"
#include "map_straight.h"

namespace libcomm {

/*!
   \brief   Channel Codec with Soft Output and same Input/Output Symbol Space.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

template <class Base, class dbl=double>
class codec_softout_flattened : public Base {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<dbl>        array1d_t;
   typedef libbase::matrix<dbl>        array2d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef codec_softout_flattened<Base,dbl> This;
public:
   /*! \name Constructors / Destructors */
   ~codec_softout_flattened() {};
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const libbase::vector< libbase::vector<double> >& ptable);

   // Codec information functions - fundamental
   libbase::size<libbase::vector> output_block_size() const
      { return libbase::size<libbase::vector>(Base::input_block_size() * \
               log2(Base::num_outputs()/Base::num_inputs())); };
   int num_outputs() const { return Base::num_inputs(); };

   // Description
   std::string description() const
      { return "Flattened " + Base::description(); };
};

// Codec operations

template <class Base, class dbl>
void codec_softout_flattened<Base,dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   // Set up mapper
   map_straight<libbase::vector> map;
   const int N = Base::num_outputs(); // From:   # enc outputs
   const int M = This::num_outputs(); // To:     # mod symbols
   const int S = Base::num_outputs(); // Unused: # tran symbols
   map.set_parameters(N, M, S);
   // Encode to a temporary space and convert
   array1i_t encwide;
   Base::encode(source, encwide);
   map.transform(encwide, encoded);
   }

template <class Base, class dbl>
void codec_softout_flattened<Base,dbl>::translate(const libbase::vector< libbase::vector<double> >& ptable)
   {
   // Set up mapper
   map_straight<libbase::vector> map;
   const int N = Base::num_outputs(); // Unused: # enc outputs
   const int M = This::num_outputs(); // From:   # mod symbols
   const int S = Base::num_outputs(); // To:     # tran symbols
   map.set_parameters(N, M, S);
   // Convert to a temporary space and translate
   libbase::vector< libbase::vector<double> > ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::translate(ptable_flat);
   }

}; // end namespace

#endif
