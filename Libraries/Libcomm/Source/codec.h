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

   \version 1.01 (5 Mar 1999)
  renamed the mem_order() function to tail_length(), which is more indicative of the true
  use of this function.

   \version 1.10 (7 Jun 1999)
  modulated sequence changed from matrix to vector, in order to simplify the implementation
  of puncturing. Now, num_symbols returns the length of the signal space vector.

   \version 1.11 (26 Oct 2001)
  added a virtual destroy function (see interleaver.h)

   \version 1.20 (4 Nov 2001)
  added a virtual function which outputs details on the codec (this was only done before
  in the construction mechanism). Added a stream << operator too.

   \version 1.21 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.

   \version 1.30 (13 Mar 2002)
  added serialization facility. Created serialize and stream << and >> functions to
  conform with the new serializer protocol, as defined in serializer 1.10. The stream
  << output function first writes the name of the derived class, then calls its
  serialize() to output the data. The name is obtained from the virtual name() function.
  The stream >> input function first gets the name from the stream, then (via
  serialize::call) creates a new object of the appropriate type and calls its serialize()
  function to get the relevant data. Also added cloning function.

   \version 1.40 (17 Mar 2002)
  added information function which returns the number of values each output symbol can
  take. Also removed the modulate/transmit/demodulate functions (which are now in the
  modulator and channel modules) and also num_symbols (since this is no longer necessary).
  Puncturing is now to be performed as a separate step, to avoid the overhead when there
  is no puncturing. Also, the num_iter function has no default value now (to make sure
  it's not overlooked). Finally, created a new function 'translate' which computes the
  necessary prabability tables for the particular codec from the probabilities of each
  modulation symbol as received from the channel. This function should be called before
  the first decode iteration for each block. Also, the input vector to encode has been
  made const.

   \version 1.41 (18 Mar 2002)
  added three informative functions, giving the number of input bits & output bits per
  frame, and the data rate for the code. These can be computed from the other information
  functions for all codes, so they have been implemented as part of the base class. Note
  that the number of bits need not be an integer (if the codes are not binary).
  Also made all information functions const. Finally, changed the input vector for
  encode() back to non-const (tail entries need to be filled in by the encoder with the
  actual values used).

   \version 1.50 (27 Mar 2002)
  removed the descriptive output() and related stream << output functions, and replaced
  them by a function description() which returns a string. This provides the same
  functionality but in a different format, so that now the only stream << output
  functions are for serialization. This should make the notation much clearer while
  also simplifying description display in objects other than streams.

   \version 1.60 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.61 (17 Apr 2008)
   - removed friend status of stream output operators

   \version 1.62 (24 Apr 2008)
   - replaced serialization support with macros

   \version 1.63 (25 Apr 2008)
   - added information function to give the required transmission symbol
     alphabet size (required by mapper); defaults to output alphabet size.

   \todo
   Change class interface to better model the actual representation of input and
   output sequences of the codec and to better separate this class from the
   modulation class.
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
   */
   virtual void encode(libbase::vector<int>& source, libbase::vector<int>& encoded) = 0;
   /*!
      \brief Receiver translation process
      \param[in] ptable Matrix representing the likelihoods of each possible
                        modulation symbol at every (modulation) timestep
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
   //! Output symbol alphabet size
   virtual int output_alphabet() const { return num_outputs(); };
   //! Length of tail in timesteps
   virtual int tail_length() const = 0;
   //! Number of iterations per decoding cycle
   virtual int num_iter() const = 0;
   // @}

   /*! \name Codec information functions - derived */
   //! Length of information sequence in bits
   double input_bits() const { return log2(num_inputs())*(block_size() - tail_length()); };
   //! Output block length in bits
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

