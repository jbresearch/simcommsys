#ifndef __codec_h
#define __codec_h

#include "config.h"
#include "matrix.h"
#include "vector.h"
#include "serializer.h"
#include "random.h"
#include <string>

namespace libcomm {

/*!
   \brief   Channel Codec Base.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo Current model assumes one symbol per timestep; this needs to change
         so we can represent multiple input/output symbols per timestep.

   \todo Change class interface to better model the actual representation of
         input and output sequences of the codec and to better separate this
         class from the modulation class.
*/

template <template<class> class C=libbase::vector>
class codec {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~codec() {};
   // @}

   /*! \name Codec operations */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) {};
   /*!
      \brief Encoding process
      \param[in] source Sequence of source symbols, one per timestep
      \param[out] encoded Sequence of output (encoded) symbols, one per timestep

      \note If the input or output symbols at every timestep represent the
            aggregation of a set of symbols, the combination/division has to
            be done externally.
   */
   virtual void encode(const C<int>& source, C<int>& encoded) = 0;
   /*!
      \brief Receiver translation process
      \param[in] ptable Likelihoods of each possible modulation symbol at every
                        (modulation) timestep

      This function computes the necessary prabability tables for the codec
      from the probabilities of each modulation symbol as received from the
      channel. This function should be called before the first decode iteration
      for each block.

      \note The number of possible modulation symbols does not necessarily
            correspond to the number of encoder output symbols, and therefore
            the number of modulation timesteps may be different from tau.
   */
   virtual void translate(const C<array1d_t>& ptable) = 0;
   /*!
      \brief Decoding process
      \param[out] decoded Most likely sequence of information symbols, one per timestep

      \note Observe that this output necessarily constitutes a hard decision.

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void decode(C<int>& decoded) = 0;
   // @}

   /*! \name Codec information functions - fundamental */
   //! Input block size in symbols
   virtual libbase::size<C> input_block_size() const = 0;
   //! Output block size in symbols
   virtual libbase::size<C> output_block_size() const = 0;
   //! Number of valid input combinations
   virtual int num_inputs() const = 0;
   //! Number of valid output combinations
   virtual int num_outputs() const = 0;
   //! Channel symbol alphabet size required for translation
   virtual int num_symbols() const { return num_outputs(); };
   //! Length of tail in timesteps
   virtual int tail_length() const = 0;
   //! Number of iterations per decoding cycle
   virtual int num_iter() const = 0;
   // @}

   /*! \name Codec information functions - derived */
   //! Equivalent length of information sequence in bits
   double input_bits() const { return log2(num_inputs())*(output_block_size() - tail_length()); };
   //! Equivalent length of output sequence in bits
   double output_bits() const { return log2(num_outputs())*output_block_size(); };
   //! Overall code rate
   double rate() const { return input_bits()/double(output_bits()); };
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(codec);
};

}; // end namespace

#endif

