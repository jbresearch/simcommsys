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
   \brief   Mapper Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines the interface for mapper classes. It integrates within
   commsys as a layer between codec and blockmodem.
*/

class mapper : protected blockprocess {
protected:
   /*! \name User-defined parameters */
   int N;   //!< Number of possible values of each encoder output
   int M;   //!< Number of possible values of each modulation symbol
   int S;   //!< Number of possible values of each translation symbol
   int tau; //!< Block size in symbols at input
   // @}

protected:
   /*! \name Helper functions */
   /*!
      \brief Determines the number of input symbols per output symbol
      \param[in]  input    Number of possible values of each input symbol
      \param[in]  output   Number of possible values of each output symbol
   */
   static int get_rate(const int input, const int output);
   // @}
   /*! \name Interface with derived classes */
   //! Setup function, called from set_parameters and set_blocksize
   virtual void setup() = 0;
   //! \copydoc transform()
   virtual void dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const = 0;
   //! \copydoc inverse()
   virtual void doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   virtual ~mapper() {};
   // @}

   /*! \name Vector mapper operations */
   /*!
      \brief Transform a sequence of encoder outputs to a channel-compatible
             alphabet
      \param[in]  in    Sequence of encoder output values
      \param[out] out   Sequence of symbols to be modulated
   */
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const;
   /*!
      \brief Inverse-transform the received symbol probabilities to a decoder-
             comaptible set
      \param[in]  pin   Table of likelihoods of possible modulation symbols
      \param[out] pout  Table of likelihoods of possible translation symbols

      \note p(i,d) is the a posteriori probability of symbol 'd' at time 'i'
   */
   void inverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const;
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) {};
   /*!
      \brief Sets input and output alphabet sizes
      \param[in]  N  Number of possible values of each encoder output
      \param[in]  M  Number of possible values of each modulation symbol
      \param[in]  S  Number of possible values of each translation symbol
   */
   void set_parameters(const int N, const int M, const int S);
   //! Sets input block size
   void set_blocksize(int tau);
   // @}

   /*! \name Informative functions */
   //! Overall mapper rate
   virtual double rate() const = 0;
   //! Gets input block size
   int input_block_size() const { return tau; };
   //! Gets output block size
   virtual int output_block_size() const { return tau; };
   // @}

   /*! \name Description */
   //! Object description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(mapper);
};

}; // end namespace

#endif
