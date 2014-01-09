/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "parametric.h"
#include "serializer.h"
#include "vector.h"
#include "matrix.h"
#include "vectorutils.h"
#include "instrumented.h"

#include "randgen.h"
#include "sigspace.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Common Channel Interface.
 * \author  Johann Briffa
 *
 * Base channel definition provides:
 * - random generator accessible by derived classes
 * - channel model defined by transmit() and receive() functions, with support
 * for insertions and deletions, as well as substitution errors.
 *
 * \todo Think out and update cloning/serialization interface
 *
 * \todo Provide default implementation for corrupt and pdf
 *
 * \todo Sort out which receive() method is really needed in the interface, and
 * which should be specific to the various channels
 */

template <class S, template <class > class C>
class basic_channel_interface : public instrumented, public parametric {
public:
   /*! \name Type definitions */
   typedef libbase::vector<S> array1s_t;
   typedef libbase::vector<double> array1d_t;
   // @}
protected:
   /*! \name Derived channel representation */
   libbase::randgen r;
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
    * \brief Pass a single symbol through the substitution channel
    * \param   s  Input (Tx) symbol
    * \return  Output (Rx) symbol
    */
   virtual S corrupt(const S& s) = 0;
   /*!
    * \brief Determine the conditional likelihood for the received symbol
    * \param   tx  Transmitted symbol being considered
    * \param   rx  Received symbol
    * \return  Likelihood \f$ P(rx|tx) \f$
    */
   virtual double pdf(const S& tx, const S& rx) const = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   virtual ~basic_channel_interface()
      {
      }
   // @}

   /*! \name Channel parameter handling */
   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }
   // @}

   /*! \name Channel functions */
   /*!
    * \brief Pass a sequence of modulation symbols through the channel
    * \param[in]  tx  Transmitted sequence of modulation symbols
    * \param[out] rx  Received sequence of modulation symbols
    *
    * Default implementation is suitable for substitution channels, and
    * performs channel-specific operation through the corrupt() override.
    *
    * \note It is possible that the \c tx and \c rx parameters actually point
    * to the same vector.
    *
    * \callergraph
    */
   virtual void transmit(const C<S>& tx, C<S>& rx) = 0;
   /*!
    * \brief Determine the per-symbol likelihoods of a sequence of received
    * modulation symbols
    * \param[in]  tx       Set of possible transmitted symbols
    * \param[in]  rx       Received sequence of modulation symbols
    * \param[out] ptable   Likelihoods corresponding to each possible
    * transmitted symbol
    *
    * Default implementation is suitable for substitution channels, and
    * performs channel-specific operation through the pdf() override.
    *
    * \note For substitution channels, this method is only suitable when the
    * modulation scheme is time-invariant
    *
    * \note Suitable for non-substitution channels, only when it is assumed
    * that 'rx' corresponds to a single transmitted symbol
    *
    * \warning Note that 'rx' assumes different meanings for substitution and
    * non-substitution channels
    */
   virtual void receive(const array1s_t& tx, const C<S>& rx,
         C<array1d_t>& ptable) const = 0;
   /*!
    * \brief Determine the per-symbol likelihoods of a sequence of received
    * modulation symbols
    * \param[in]  tx       Set of possible transmitted symbols at each timestep
    * \param[in]  rx       Received sequence of modulation symbols
    * \param[out] ptable   Likelihoods corresponding to each possible
    * transmitted symbol
    *
    * Default implementation is suitable for substitution channels, and
    * performs channel-specific operation through the pdf() override.
    *
    * \note For substitution channels, this method is suitable for time-variant
    * modulation schemes
    *
    * \note Not suitable for non-substitution channels
    */
   virtual void receive(const C<array1s_t>& tx, const C<S>& rx,
         C<array1d_t>& ptable) const = 0;
   /*!
    * \brief Determine the likelihood of a sequence of received modulation
    * symbols, given a particular transmitted sequence
    * \param[in]  tx       Transmitted sequence being considered
    * \param[in]  rx       Received sequence of modulation symbols
    * \return              Likelihood \f$ P(rx|tx) \f$
    *
    * \note Suitable for non-substitution channels, where length of 'tx' and
    * 'rx' are not necessarily the same
    */
   virtual double receive(const C<S>& tx, const C<S>& rx) const = 0;
   /*!
    * \brief Determine the likelihood of a sequence of received modulation
    * symbols, given a particular transmitted symbol
    * \param[in]  tx       Transmitted symbol being considered
    * \param[in]  rx       Received sequence of modulation symbols
    * \return              Likelihood \f$ P(rx|tx) \f$
    *
    * \note Suitable for non-substitution channels, where length of 'rx' is
    * not necessarily equal to 1
    */
   virtual double receive(const S& tx, const C<S>& rx) const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 * \brief   Common Channel Base.
 * \author  Johann Briffa
 *
 * Templated common channel base. This extra level is required to allow partial
 * specialization of the container.
 */

template <class S, template <class > class C>
class basic_channel : public basic_channel_interface<S, C> {
};

/*!
 * \brief   Common Channel Base Specialization.
 * \author  Johann Briffa
 *
 * Templated common channel base. Partial specialization for vector container.
 */

template <class S>
class basic_channel<S, libbase::vector> : public basic_channel_interface<S,
      libbase::vector> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<S> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
public:
   void transmit(const array1s_t& tx, array1s_t& rx);
   void
   receive(const array1s_t& tx, const array1s_t& rx, array1vd_t& ptable) const;
   void
   receive(const array1vs_t& tx, const array1s_t& rx, array1vd_t& ptable) const;
   double receive(const array1s_t& tx, const array1s_t& rx) const;
   double receive(const S& tx, const array1s_t& rx) const;
};

// channel functions

template <class S>
void basic_channel<S, libbase::vector>::transmit(const array1s_t& tx,
      array1s_t& rx)
   {
   // Initialize results vector
   rx.init(tx.size());
   // Corrupt the modulation symbols (simulate the channel)
   for (int i = 0; i < tx.size(); i++)
      rx(i) = this->corrupt(tx(i));
   }

template <class S>
void basic_channel<S, libbase::vector>::receive(const array1s_t& tx,
      const array1s_t& rx, array1vd_t& ptable) const
   {
   // Compute sizes
   const int tau = rx.size();
   const int M = tx.size();
   // Initialize results vector
   libbase::allocate(ptable, tau, M);
   // Work out the probabilities of each possible signal
   for (int t = 0; t < tau; t++)
      for (int x = 0; x < M; x++)
         ptable(t)(x) = this->pdf(tx(x), rx(t));
   }

template <class S>
void basic_channel<S, libbase::vector>::receive(const array1vs_t& tx,
      const array1s_t& rx, array1vd_t& ptable) const
   {
   // Compute sizes
   const int tau = rx.size();
   assert(tx.size() == tau);
   assert(tau > 0);
   const int M = tx(0).size();
   // Initialize results vector
   libbase::allocate(ptable, tau, M);
   // Work out the probabilities of each possible signal
   for (int t = 0; t < tau; t++)
      {
      assert(tx(t).size() == M);
      for (int x = 0; x < M; x++)
         ptable(t)(x) = this->pdf(tx(t)(x), rx(t));
      }
   }

template <class S>
double basic_channel<S, libbase::vector>::receive(const array1s_t& tx,
      const array1s_t& rx) const
   {
   // Compute sizes
   const int tau = rx.size();
   // This implementation only works for substitution channels
   assert(tx.size() == tau);
   // Work out the combined probability of the sequence
   double p = 1;
   for (int t = 0; t < tau; t++)
      p *= this->pdf(tx(t), rx(t));
   return p;
   }

template <class S>
double basic_channel<S, libbase::vector>::receive(const S& tx,
      const array1s_t& rx) const
   {
   // This implementation only works for substitution channels
   assert(rx.size() == 1);
   // Work out the probability of receiving the particular symbol
   return this->pdf(tx, rx(0));
   }

/*!
 * \brief   Common Channel Base Specialization.
 * \author  Johann Briffa
 *
 * Templated common channel base. Partial specialization for matrix container.
 */

template <class S>
class basic_channel<S, libbase::matrix> : public basic_channel_interface<S,
      libbase::matrix> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<S> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::matrix<S> array2s_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::matrix<array1d_t> array2vd_t;
   // @}
public:
   void transmit(const array2s_t& tx, array2s_t& rx)
      {
      // Initialize results vector
      rx.init(tx.size());
      // Corrupt the modulation symbols (simulate the channel)
      for (int i = 0; i < tx.size().rows(); i++)
         for (int j = 0; j < tx.size().cols(); j++)
            rx(i, j) = this->corrupt(tx(i, j));
      }
   void receive(const array1s_t& tx, const array2s_t& rx, array2vd_t& ptable) const
      {
      // Compute sizes
      const int M = tx.size();
      // Initialize results vector
      libbase::allocate(ptable, rx.size().rows(), rx.size().cols(), M);
      // Work out the probabilities of each possible signal
      for (int i = 0; i < rx.size().rows(); i++)
         for (int j = 0; j < rx.size().cols(); j++)
            for (int x = 0; x < M; x++)
               ptable(i, j)(x) = this->pdf(tx(x), rx(i, j));
      }
   void receive(const array2vs_t& tx, const array2s_t& rx, array2vd_t& ptable) const
      {
      // Compute sizes
      assert(tx.size() == rx.size());
      assert(tx.size() > 0);
      const int M = tx(0, 0).size();
      // Initialize results vector
      libbase::allocate(ptable, rx.size().rows(), rx.size().cols(), M);
      // Work out the probabilities of each possible signal
      for (int i = 0; i < rx.size().rows(); i++)
         for (int j = 0; j < rx.size().cols(); j++)
            {
            assert(tx(i, j).size() == M);
            for (int x = 0; x < M; x++)
               ptable(i, j)(x) = this->pdf(tx(i, j)(x), rx(i, j));
            }
      }
   double receive(const array2s_t& tx, const array2s_t& rx) const
      {
      // This implementation only works for substitution channels
      assert(tx.size().rows() == rx.size().rows());
      assert(tx.size().cols() == rx.size().cols());
      // Work out the combined probability of the sequence
      double p = 1;
      for (int i = 0; i < rx.size().rows(); i++)
         for (int j = 0; j < rx.size().cols(); j++)
            p *= this->pdf(tx(i, j), rx(i, j));
      return p;
      }
   double receive(const S& tx, const array2s_t& rx) const
      {
      // This implementation only works for substitution channels
      assert(rx.size() == 1);
      // Work out the probability of receiving the particular symbol
      return this->pdf(tx, rx(0, 0));
      }
};

/*!
 * \brief   Channel Base.
 * \author  Johann Briffa
 *
 * Templated base channel model.
 */

template <class S, template <class > class C = libbase::vector>
class channel : public basic_channel<S, C> , public libbase::serializable {
   // Serialization Support
DECLARE_BASE_SERIALIZER(channel)
};

/*!
 * \brief   Signal-Space Channel.
 * \author  Johann Briffa
 *
 * Class specialization including elements specific to the signal-space
 * channel model.
 */

template <template <class > class C>
class channel<sigspace, C> : public basic_channel<sigspace, C> ,
      public libbase::serializable {
private:
   /*! \name User-defined parameters */
   double snr_db; //!< Equal to \f$ 10 \log_{10} ( \frac{E_b}{N_0} ) \f$
   // @}
   /*! \name Internal representation */
   double Eb; //!< Average signal energy per information bit \f$ E_b \f$
   double No; //!< Half the noise energy/modulation symbol for a normalised signal \f$ N_0 \f$.
   // @}
private:
   /*! \name Internal functions */
   void compute_noise()
      {
      No = 0.5 * pow(10.0, -snr_db / 10.0);
      // call derived class handle
      compute_parameters(Eb, No);
      }
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
    * \brief Determine channel-specific parameters based on given SNR
    *
    * \note \f$ E_b \f$ is fixed by the overall modulation and coding system.
    * The simulator determines \f$ N_0 \f$ according to the given SNR
    * (assuming unit signal energy), so that the actual band-limited
    * noise energy is given by \f$ E_b N_0 \f$.
    */
   virtual void compute_parameters(const double Eb, const double No)
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   channel()
      {
      Eb = 1;
      set_parameter(0);
      }
   // @}

   /*! \name Channel parameter handling */
   //! Set the bit-equivalent signal energy
   void set_eb(const double Eb)
      {
      channel::Eb = Eb;
      compute_noise();
      }
   //! Set the normalized noise energy
   void set_no(const double No)
      {
      snr_db = -10.0 * log10(2 * No);
      compute_noise();
      }
   //! Get the bit-equivalent signal energy
   double get_eb() const
      {
      return Eb;
      }
   //! Get the normalized noise energy
   double get_no() const
      {
      return No;
      }
   //! Set the signal-to-noise ratio
   void set_parameter(const double snr_db)
      {
      channel::snr_db = snr_db;
      compute_noise();
      }
   //! Get the signal-to-noise ratio
   double get_parameter() const
      {
      return snr_db;
      }
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(channel)
};

} // end namespace

#endif
