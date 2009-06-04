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
   public codec<libbase::matrix> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>        array1d_t;
   // @}
private:
   /*! \name Internal representation */
   base_codec base;
   // @}
public:
   /*! \name Constructors / Destructors */
   ~codec_reshaped() {};
   // @}

   // Codec operations
   void seedfrom(libbase::random& r) { base.seedfrom(r); };
   void encode(const libbase::matrix<int>& source, libbase::matrix<int>& encoded)
      {
      libbase::vector<int> source_v = source;
      libbase::vector<int> encoded_v;
      base.encode(source_v, encoded_v);
      encoded = encoded_v;
      }
   void translate(const libbase::matrix<array1d_t>& ptable)
      {
      libbase::vector<array1d_t> ptable_v = ptable;
      base.translate(ptable_v);
      }
   void decode(libbase::matrix<int>& decoded)
      {
      libbase::vector<int> decoded_v;
      base.decode(decoded_v);
      decoded = decoded_v;
      }

   // Codec information functions - fundamental
   libbase::size_type<libbase::matrix> input_block_size() const
      {
      // Inherit sizes
      const int N = base.input_block_size();
      return libbase::size_type<libbase::matrix>(N,1);
      };
   libbase::size_type<libbase::matrix> output_block_size() const
      {
      // Inherit sizes
      const int N = base.output_block_size();
      return libbase::size_type<libbase::matrix>(N,1);
      };
   int num_inputs() const { return base.num_inputs(); };
   int num_outputs() const { return base.num_outputs(); };
   int num_symbols() const { return base.num_symbols(); };
   int tail_length() const { return base.tail_length(); };
   int num_iter() const { return base.num_iter(); };

   // Description
   std::string description() const
      { return "Reshaped " + base.description(); };

   // Serialization Support
   DECLARE_SERIALIZER(codec_reshaped);
};

}; // end namespace

#endif
