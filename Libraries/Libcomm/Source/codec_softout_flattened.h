#ifndef __codec_softout_flattened_h
#define __codec_softout_flattened_h

#include "config.h"
#include "map_straight.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show mapping block sizes
// 3 - Show soft-output for encoded symbols
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 \brief   Channel Codec with Soft Output and same Input/Output Symbol Space.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

template <class base_codec_softout, class dbl = double>
class codec_softout_flattened : public base_codec_softout {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef base_codec_softout Base;
   typedef codec_softout_flattened<base_codec_softout, dbl> This;
private:
   /*! \name Internal functions */
   template <class D> void init(mapper<libbase::vector, D>& map) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   ~codec_softout_flattened()
      {
      }
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void init_decoder(const array1vd_t& ptable);
   void init_decoder(const array1vd_t& ptable, const array1vd_t& app);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // Inherit sizes
      const int N = Base::output_block_size();
      const int n = Base::num_outputs();
      const int k = Base::num_inputs();
      return libbase::size_type<libbase::vector>(
            int(round(N * log(n) / log(k))));
      }
   int num_outputs() const
      {
      return Base::num_inputs();
      }

   // Description
   std::string description() const
      {
      return "Flattened " + Base::description();
      }
};

// Initialization

template <class base_codec_softout, class dbl>
template <class D>
void codec_softout_flattened<base_codec_softout, dbl>::init(mapper<
      libbase::vector, D>& map) const
   {
   // Set up mapper
   const int N = Base::num_outputs(); // # enc outputs
   const int M = This::num_outputs(); // # mod symbols
   const int S = Base::num_outputs(); // # tran symbols
   map.set_parameters(N, M, S);
   map.set_blocksize(Base::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG: mapper setup from "
   << map.input_block_size() << "x" << N << " to "
   << map.output_block_size() << "x" << M << " to "
   << map.input_block_size() << "x" << S << " symbols\n";
#endif
   }

// Codec operations

template <class base_codec_softout, class dbl>
void codec_softout_flattened<base_codec_softout, dbl>::encode(
      const array1i_t& source, array1i_t& encoded)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Encode to a temporary space and convert
   array1i_t encwide;
   Base::encode(source, encwide);
   map.transform(encwide, encoded);
   }

template <class base_codec_softout, class dbl>
void codec_softout_flattened<base_codec_softout, dbl>::init_decoder(
      const array1vd_t& ptable)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Convert to a temporary space and translate
   array1vd_t ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::init_decoder(ptable_flat);
   }

template <class base_codec_softout, class dbl>
void codec_softout_flattened<base_codec_softout, dbl>::init_decoder(
      const array1vd_t& ptable, const array1vd_t& app)
   {
   map_straight<libbase::vector, dbl> map;
   init(map);
   // Convert to a temporary space and translate
   array1vd_t ptable_flat;
   map.inverse(ptable, ptable_flat);
   Base::init_decoder(ptable_flat, app);
   }

template <class base_codec_softout, class dbl>
void codec_softout_flattened<base_codec_softout, dbl>::softdecode(
      array1vd_t& ri, array1vd_t& ro)
   {
   // Decode to a temporary space
   array1vd_t ro_wide;
   Base::softdecode(ri, ro_wide);
#if DEBUG>=3
   array1i_t dec;
   This::hard_decision(ro_wide,dec);
   libbase::trace << "DEBUG (csf): ro_wide = ";
   dec.serialize(libbase::trace, ' ');
#endif
   // Allocate space for results
   ro.init(This::output_block_size());
   for (int t = 0; t < This::output_block_size(); t++)
      ro(t).init(This::num_outputs());
   // Initialize
   ro = 0.0;
   // Convert
   const int N = Base::num_outputs(); // # enc outputs
   const int M = This::num_outputs(); // # mod symbols
   const int s1 = mapper<>::get_rate(M, N);
   for (int t = 0; t < ro_wide.size(); t++)
      for (int x = 0; x < ro_wide(t).size(); x++)
         for (int i = 0, thisx = x; i < s1; i++, thisx /= M)
            ro(t * s1 + i)(thisx % M) += ro_wide(t)(x);
#if DEBUG>=3
   This::hard_decision(ro,dec);
   libbase::trace << "DEBUG (csf): ro = ";
   dec.serialize(libbase::trace, ' ');
#endif
   }

} // end namespace

#endif
