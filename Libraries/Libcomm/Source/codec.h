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

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Change class interface to better model the actual representation of
         input and output sequences of the codec and to better separate this
         class from the modulation class.
*/

class codec {
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
      \param[in,out] source Sequence of source symbols, one per timestep
      \param[in,out] encoded Sequence of output (encoded) symbols, one per timestep

      \note If the input or output symbols at every timestep represent the
            aggregation of a set of symbols, the combination/division has to
            be done externally.

      \note Source is non-const; this is necessary to allow the codec to set
            values actually used in tail.
   */
   virtual void encode(libbase::vector<int>& source, libbase::vector<int>& encoded) = 0;
   /*!
      \brief Receiver translation process
      \param[in] ptable Matrix representing the likelihoods of each possible
                        modulation symbol at every (modulation) timestep

      This function computes the necessary prabability tables for the codec
      from the probabilities of each modulation symbol as received from the
      channel. This function should be called before the first decode iteration
      for each block.

      \note The number of possible modulation symbols does not necessarily
            correspond to the number of encoder output symbols, and therefore
            the number of modulation timesteps may be different from tau.
   */
   virtual void translate(const libbase::matrix<double>& ptable) = 0;
   /*!
      \brief Encoding process
      \param[out] decoded Most likely sequence of information symbols, one per timestep

      \note Observe that this output necessarily constitutes a hard decision.
            Also, each call to decode will perform a single iteration (with
            respect to num_iter).
   */
   virtual void decode(libbase::vector<int>& decoded) = 0;
   // @}

   /*! \name Codec information functions - fundamental */
   //! Block size (input length in timesteps)
   virtual int block_size() const = 0;
   //! Number of valid input combinations
   virtual int num_inputs() const = 0;
   //! Number of valid output combinations
   virtual int num_outputs() const = 0;
   //! Output symbol alphabet size (required by mapper)
   virtual int output_alphabet() const { return num_outputs(); };
   //! Length of tail in timesteps
   virtual int tail_length() const = 0;
   //! Number of iterations per decoding cycle
   virtual int num_iter() const = 0;
   // @}

   /*! \name Codec information functions - derived */
   //! Equivalent length of information sequence in bits
   double input_bits() const { return log2(num_inputs())*(block_size() - tail_length()); };
   //! Equivalent length of output sequence in bits
   double output_bits() const { return log2(num_outputs())*block_size(); };
   //! Overall code rate
   double rate() const { return input_bits()/output_bits(); };
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(codec)
};

}; // end namespace

#endif

