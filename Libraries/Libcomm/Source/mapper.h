#ifndef __mapper_h
#define __mapper_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "serializer.h"
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

   \version 1.00 (21-24 Apr 2008)
   - Defines interface for mapper classes.
   - Defined a straight symbol mapper with:
      * forward transform from modulator
      * inverse transform from the various codecs.

   \todo
   - Integrate within commsys as a layer between codec and modulator.
*/

class mapper {
   static const libbase::serializer shelper;
   static void* create() { return new mapper; };
public:
   /*! \name Constructors / Destructors */
   virtual ~mapper() {};
   // @}
   /*! \name Serialization Support */
   virtual mapper *clone() const { return new mapper(*this); };
   virtual const char* name() const { return shelper.name(); };
   // @}

   /*! \name Vector mapper operations */
   /*!
      \brief Transform a sequence of encoder outputs to a channel-compatible alphabet
      \param[in]  N        The number of possible values of each encoded element
      \param[in]  encoded  Sequence of values to be modulated
      \param[in]  M        The number of possible values of each transmitted element
      \param[out] tx       Sequence of symbols corresponding to the given input

      \todo Remove parameters N and M, replacing 'int' type for encoded vector with
            something that also encodes the number of symbols in the alphabet
   */
   virtual void transform(const int N, const libbase::vector<int>& encoded, const int M, libbase::vector<int>& tx);
   /*!
      \brief Inverse-transform the received symbol probabilities to a decoder-comaptible set
      \param[in]  pin      Table of likelihoods of possible modulation symbols
      \param[in]  N        The number of possible values of each encoder element
                           (this is what the encoder would like)
      \param[out] pout     Table of likelihoods of possible encoder symbols
      
      \note \c pxxx(i,d) \c is the a posteriori probability of symbol 'd' at time 'i'
   */
   virtual void inverse(const libbase::matrix<double>& pin, const int N, libbase::matrix<double>& pout);
   // @}

   /*! \name Setup functions */
   //! Reset function for random generator
   virtual void seed(libbase::int32u const s) {};
   // @}

   /*! \name Informative functions */
   //! Overall mapper rate
   virtual double rate() const { return 1; };
   // @}

   /*! \name Description & Serialization */
   //! Object description output
   virtual std::string description() const;
   //! Object serialization ouput
   virtual std::ostream& serialize(std::ostream& sout) const  { return sout; };
   //! Object serialization input
   virtual std::istream& serialize(std::istream& sin) { return sin; };
   // @}
};

/*! \name Serialization */

std::ostream& operator<<(std::ostream& sout, const mapper* x);
std::istream& operator>>(std::istream& sin, mapper*& x);

// @}

}; // end namespace

#endif
