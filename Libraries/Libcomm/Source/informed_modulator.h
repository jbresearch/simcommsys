#ifndef __informed_modulator_h
#define __informed_modulator_h

#include "blockmodem.h"

namespace libcomm {

/*!
   \brief   Informed Modulator Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Defines common interface for informed blockmodem classes. An informed
   blockmodem is one which can use a-priori symbol probabilities during the
   demodulation stage. In general, such a blockmodem may be used in an iterative
   loop with the channel codec.

   This interface is a superset of the regular blockmodem, defining two new
   demodulation methods (atomic and vector) that make use of prior information.
*/

template <class S>
class informed_modulator : public blockmodem<S> {
protected:
   /*! \name Interface with derived classes */
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S>& chan, const libbase::vector<S>& rx, const libbase::vector< libbase::vector<double> >& app, libbase::vector< libbase::vector<double> >& ptable) = 0;
   // @}

public:
   /*! \name Atomic modem operations */
   /*!
      \brief Demodulate a single time-step
      \param[in]  signal   Received signal
      \param[in]  app      Table of a-priori likelihoods of possible
                           transmitted symbols
      \return  Index corresponding symbol that is closest to the received signal
   */
   virtual const int demodulate(const S& signal, const libbase::vector<double>& app) const = 0;
   // @}

   /*! \name Vector modem operations */
   /*!
      \brief Demodulate a sequence of time-steps
      \param[in]  chan     The channel model (used to obtain likelihoods)
      \param[in]  rx       Sequence of received symbols
      \param[in]  app      Table of a-priori likelihoods of possible
                           transmitted symbols at every time-step
      \param[out] ptable   Table of likelihoods of possible transmitted symbols

      \note \c ptable(i,d) \c is the a posteriori probability of having transmitted
            symbol 'd' at time 'i'

      \note This function is non-const, to support time-variant modulation
            schemes such as DM inner codes.
   */
   void demodulate(const channel<S>& chan, const libbase::vector<S>& rx, const libbase::matrix<double>& app, libbase::matrix<double>& ptable);
   // @}
};

}; // end namespace

#endif
