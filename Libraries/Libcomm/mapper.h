#ifndef __mapper_h
#define __mapper_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "serializer.h"
#include "random.h"
#include "blockprocess.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Mapper Interface.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * This class defines the interface for mapper classes. It integrates within
 * commsys as a layer between codec and blockmodem.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class mapper : public blockprocess {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef mapper<C, dbl> This;

protected:
   /*! \name User-defined parameters */
   int N; //!< Number of possible values of each encoder output
   int M; //!< Number of possible values of each modulation symbol
   int S; //!< Number of possible values of each translation symbol
   libbase::size_type<C> size; //!< Input block size in symbols
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Setup function, called from set_parameters and set_blocksize
   virtual void setup() = 0;
   //! \copydoc transform()
   virtual void dotransform(const C<int>& in, C<int>& out) const = 0;
   //! \copydoc inverse()
   virtual void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~mapper()
      {
      }
   // @}

   /*! \name Helper functions */
   /*!
    * \brief Determines the number of input symbols per output symbol
    * \param[in]  input    Number of possible values of each input symbol
    * \param[in]  output   Number of possible values of each output symbol
    */
   static int get_rate(const int input, const int output);
   // @}

   /*! \name Vector mapper operations */
   /*!
    * \brief Transform a sequence of encoder outputs to a channel-compatible
    * alphabet
    * \param[in]  in    Sequence of encoder output values
    * \param[out] out   Sequence of symbols to be modulated
    */
   void transform(const C<int>& in, C<int>& out) const;
   /*!
    * \brief Inverse-transform the received symbol probabilities to a decoder-
    * comaptible set
    * \param[in]  pin   Table of likelihoods of possible modulation symbols
    * \param[out] pout  Table of likelihoods of possible translation symbols
    * 
    * \note p(i,d) is the a posteriori probability of symbol 'd' at time 'i'
    */
   void inverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   /*!
    * \brief Sets input and output alphabet sizes
    * \param[in]  N  Number of possible values of each encoder output
    * \param[in]  M  Number of possible values of each modulation symbol
    * \param[in]  S  Number of possible values of each translation symbol
    */
   void set_parameters(const int N, const int M, const int S);
   //! Sets input block size
   void set_blocksize(libbase::size_type<C> size)
      {
      assert(size > 0);
      This::size = size;
      setup();
      }
   // @}

   /*! \name Informative functions */
   //! Overall mapper rate
   virtual double rate() const = 0;
   //! Gets input block size
   libbase::size_type<C> input_block_size() const
      {
      return size;
      }
   //! Gets output block size
   virtual libbase::size_type<C> output_block_size() const
      {
      return size;
      }
   // @}

   /*! \name Description */
   //! Object description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(mapper)
};

} // end namespace

#endif
