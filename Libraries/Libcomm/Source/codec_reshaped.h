#ifndef __codec_reshaped_h
#define __codec_reshaped_h

#include "config.h"
#include "codec.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
   \brief   Channel Codec with matrix container from vector container.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

template <class base_codec>
class codec_reshaped :
   public codec<libbase::matrix>,
   public base_codec {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>        array1d_t;
   // @}
private:
   // Shorthand for containers
   typedef libbase::matrix C;
   typedef libbase::vector V;
   // Shorthand for class hierarchy
   typedef base_codec Base;
   typedef codec_reshaped<base_codec> This;
public:
   /*! \name Constructors / Destructors */
   ~codec_reshaped() {};
   // @}

   // Codec operations
   void encode(const C<int>& source, C<int>& encoded)
      {
      V<int> source_v = source;
      V<int> encoded_v;
      Base::encode(source_v, encoded_v);
      encoded = encoded_v;
      }
   void translate(const C<array1d_t>& ptable)
      {
      V<array1d_t> ptable_v = ptable;
      Base::translate(ptable_v);
      }
   void decode(C<int>& decoded)
      {
      V<int> decoded_v;
      Base::decode(decoded_v);
      decoded = decoded_v;
      }

   // Codec information functions - fundamental
   libbase::size<C> input_block_size() const
      {
      // Inherit sizes
      const int N = Base::input_block_size();
      return libbase::size<C>(N,1);
      };
   libbase::size<C> output_block_size() const
      {
      // Inherit sizes
      const int N = Base::output_block_size();
      return libbase::size<C>(N,1);
      };

   // Description
   std::string description() const
      { return "Reshaped " + Base::description(); };
};

}; // end namespace

#endif
