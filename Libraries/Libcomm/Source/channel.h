#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "parametric.h"
#include "serializer.h"
#include "vector.h"
#include "matrix.h"

#include "randgen.h"
#include "sigspace.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Common Channel Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Base channel definition provides:
   - random generator accessible by derived classes
   - channel model defined by transmit() and receive() functions, with support
     for insertions and deletions, as well as substitution errors.

   \todo Think out and update cloning/serialization interface

   \todo Provide default implementation for corrupt and pdf

   \todo Sort out which receive() method is really needed in the interface, and
         which should be specific to the various channels
*/

template <class S, template<class> class C>
class basic_channel_interface : public parametric {
protected:
   /*! \name Derived channel representation */
   libbase::randgen  r;
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
      \brief Pass a single symbol through the substitution channel
      \param   s  Input (Tx) symbol
      \return  Output (Rx) symbol
   */
   virtual S corrupt(const S& s) = 0;
   /*!
      \brief Determine the conditional likelihood for the received symbol
      \param   tx  Transmitted symbol being considered
      \param   rx  Received symbol
      \return  Likelihood \f$ P(rx|tx) \f$
   */
   virtual double pdf(const S& tx, const S& rx) const = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   virtual ~basic_channel_interface() {};
   // @}

   /*! \name Channel parameter handling */
   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };
   // @}

   /*! \name Channel functions */
   /*!
      \brief Pass a sequence of modulation symbols through the channel
      \param[in]  tx  Transmitted sequence of modulation symbols
      \param[out] rx  Received sequence of modulation symbols

      Default implementation is suitable for substitution channels, and
      performs channel-specific operation through the corrupt() override.

      \note It is possible that the \c tx and \c rx parameters actually point
            to the same vector.

      \callergraph
   */
   virtual void transmit(const C<S>& tx, C<S>& rx) = 0;
   /*!
      \brief Determine the per-symbol likelihoods of a sequence of received
             modulation symbols corresponding to one transmission step
      \param[in]  tx       Set of possible transmitted symbols
      \param[in]  rx       Received sequence of modulation symbols
      \param[out] ptable   Likelihoods corresponding to each possible
                           transmitted symbol

      Default implementation is suitable for substitution channels, and
      performs channel-specific operation through the pdf() override.

      \callergraph
   */
   virtual void receive(const C<S>& tx, const C<S>& rx, C< libbase::vector<double> >& ptable) const = 0;
   /*!
      \brief Determine the likelihood of a sequence of received modulation
             symbols, given a particular transmitted sequence
      \param[in]  tx       Transmitted sequence being considered
      \param[in]  rx       Received sequence of modulation symbols
      \return              Likelihood \f$ P(rx|tx) \f$

      \callergraph
   */
   virtual double receive(const C<S>& tx, const C<S>& rx) const = 0;
   /*!
      \brief Determine the likelihood of a sequence of received modulation
             symbols, given a particular transmitted symbol
      \param[in]  tx       Transmitted symbol being considered
      \param[in]  rx       Received sequence of modulation symbols
      \return              Likelihood \f$ P(rx|tx) \f$

      \callergraph
   */
   virtual double receive(const S& tx, const C<S>& rx) const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
   \brief   Common Channel Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Templated common channel base. This extra level is required to allow partial
   specialization of the container.
*/

template <class S, template<class> class C>
class basic_channel : public basic_channel_interface<S,C> {
};

/*!
   \brief   Common Channel Base Specialization.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Templated common channel base. Partial specialization for vector container.
*/

template <class S>
class basic_channel<S,libbase::vector> : public basic_channel_interface<S,libbase::vector> {
public:
   void transmit(const libbase::vector<S>& tx, libbase::vector<S>& rx);
   void receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx, libbase::vector< libbase::vector<double> >& ptable) const;
   double receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx) const;
   double receive(const S& tx, const libbase::vector<S>& rx) const;
};

// channel functions

template <class S>
void basic_channel<S,libbase::vector>::transmit(const libbase::vector<S>& tx, libbase::vector<S>& rx)
   {
   // Initialize results vector
   rx.init(tx);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<tx.size(); i++)
      rx(i) = corrupt(tx(i));
   }

template <class S>
void basic_channel<S,libbase::vector>::receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx, libbase::vector< libbase::vector<double> >& ptable) const
   {
   // Compute sizes
   const int tau = rx.size();
   const int M = tx.size();
   // Initialize results vector
   ptable.init(tau);
   for(int t=0; t<tau; t++)
      ptable(t).init(M);
   // Work out the probabilities of each possible signal
   for(int t=0; t<tau; t++)
      for(int x=0; x<M; x++)
         ptable(t)(x) = pdf(tx(x), rx(t));
   }

template <class S>
double basic_channel<S,libbase::vector>::receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx) const
   {
   // Compute sizes
   const int tau = rx.size();
   // This implementation only works for substitution channels
   assert(tx.size() == tau);
   // Work out the combined probability of the sequence
   double p = 1;
   for(int t=0; t<tau; t++)
      p *= pdf(tx(t), rx(t));
   return p;
   }

template <class S>
double basic_channel<S,libbase::vector>::receive(const S& tx, const libbase::vector<S>& rx) const
   {
   // This implementation only works for substitution channels
   assert(rx.size() == 1);
   // Work out the probability of receiving the particular symbol
   return pdf(tx, rx(0));
   }

/*!
   \brief   Channel Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Templated base channel model.
*/

template <class S, template<class> class C=libbase::vector>
class channel : public basic_channel<S,C> {
   // Serialization Support
   DECLARE_BASE_SERIALIZER(channel);
};

/*!
   \brief   Signal-Space Channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Class specialization including elements specific to the signal-space
   channel model.
*/

template <>
class channel<sigspace> : public basic_channel<sigspace,libbase::vector> {
private:
   /*! \name User-defined parameters */
   double   snr_db;  //!< Equal to \f$ 10 \log_{10} ( \frac{E_b}{N_0} ) \f$
   // @}
   /*! \name Internal representation */
   double   Eb;      //!< Average signal energy per information bit \f$ E_b \f$
   double   No;      //!< Half the noise energy/modulation symbol for a normalised signal \f$ N_0 \f$.
   // @}
private:
   /*! \name Internal functions */
   void compute_noise();
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
      \brief Determine channel-specific parameters based on given SNR

      \note \f$ E_b \f$ is fixed by the overall modulation and coding system.
            The simulator determines \f$ N_0 \f$ according to the given SNR
            (assuming unit signal energy), so that the actual band-limited
            noise energy is given by \f$ E_b N_0 \f$.
   */
   virtual void compute_parameters(const double Eb, const double No) {};
   // @}
public:
   /*! \name Constructors / Destructors */
   channel();
   // @}

   /*! \name Channel parameter handling */
   //! Set the bit-equivalent signal energy
   void set_eb(const double Eb);
   //! Set the normalized noise energy
   void set_no(const double No);
   //! Get the bit-equivalent signal energy
   double get_eb() const { return Eb; };
   //! Get the normalized noise energy
   double get_no() const { return No; };
   //! Set the signal-to-noise ratio
   void set_parameter(const double snr_db);
   //! Get the signal-to-noise ratio
   double get_parameter() const { return snr_db; };

   // Serialization Support
   DECLARE_BASE_SERIALIZER(channel);
};

}; // end namespace

#endif
