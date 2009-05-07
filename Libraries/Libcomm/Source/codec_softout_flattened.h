#ifndef __codec_softout_flattened_h
#define __codec_softout_flattened_h

#include "config.h"
#include "map_straight.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show mapping block sizes
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 2
#endif

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
private:
   /*! \name Internal functions */
   template <class D> void init(mapper<libbase::vector,D>& map) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   ~codec_softout_flattened() {};
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const libbase::vector< libbase::vector<double> >& ptable);
   void translate(const array1vd_t& ptable, const array1vd_t& app);

   // Codec information functions - fundamental
   libbase::size<libbase::vector> output_block_size() const
      {
      // Inherit sizes
      const int N = Base::output_block_size();
      const int n = Base::num_outputs();
      const int k = Base::num_inputs();
      return libbase::size<libbase::vector>(int(round(N * log(n)/log(k))));
      };
   int num_outputs() const { return Base::num_inputs(); };

   // Description
   std::string description() const
      { return "Flattened " + Base::description(); };
};

// Initialization

template <class Base, class dbl>
template <class D>
void codec_softout_flattened<Base,dbl>::init(mapper<libbase::vector,D>& map) const
   {
   // Set up mapper
   const int N = Base::num_outputs(); // From:   # enc outputs
   const int M = This::num_outputs(); // To:     # mod symbols
   const int S = Base::num_outputs(); // Unused: # tran symbols
   map.set_parameters(N, M, S);
   map.set_blocksize(Base::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG: flat encode setup from " \
      << map.input_block_size() << "x" << N << " to " \
      << map.output_block_size() << "x" << M << " symbols\n";
   libbase::trace << "DEBUG: flat translate setup from " \
      << map.output_block_size() << "x" << M << " to " \
      << map.input_block_size() << "x" << S << " symbols\n";
#endif
   }

// Codec operations

template <class Base, class dbl>
void codec_softout_flattened<Base,dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   map_straight<libbase::vector,dbl> map;
   init(map);
   // Encode to a temporary space and convert
   array1i_t encwide;
   Base::encode(source, encwide);
   map.transform(encwide, encoded);
   }

template <class Base, class dbl>
void codec_softout_flattened<Base,dbl>::translate(const libbase::vector< libbase::vector<double> >& ptable)
   {
   map_straight<libbase::vector,double> map;
   init(map);
   // Convert to a temporary space and translate
   libbase::vector< libbase::vector<double> > ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::translate(ptable_flat);
   }

template <class Base, class dbl>
void codec_softout_flattened<Base,dbl>::translate(const array1vd_t& ptable, const array1vd_t& app)
   {
   map_straight<libbase::vector,dbl> map;
   init(map);
   // Convert to a temporary space and translate
   array1vd_t ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::translate(ptable_flat,app);
   }

}; // end namespace

#endif
